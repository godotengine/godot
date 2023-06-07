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

// LZVN low-level encoder

#include "lzvn_encode_base.h"

#if defined(_MSC_VER) && !defined(__clang__)
#  define restrict __restrict
#endif

// ===============================================================
// Coarse/fine copy, non overlapping buffers

/*! @abstract Copy at least \p nbytes bytes from \p src to \p dst, by blocks
 * of 8 bytes (may go beyond range). No overlap.
 * @return \p dst + \p nbytes. */
static inline unsigned char *lzvn_copy64(unsigned char *restrict dst,
                                         const unsigned char *restrict src,
                                         size_t nbytes) {
  for (size_t i = 0; i < nbytes; i += 8)
    store8(dst + i, load8(src + i));
  return dst + nbytes;
}

/*! @abstract Copy exactly \p nbytes bytes from \p src to \p dst (respects range).
 * No overlap.
 * @return \p dst + \p nbytes. */
static inline unsigned char *lzvn_copy8(unsigned char *restrict dst,
                                        const unsigned char *restrict src,
                                        size_t nbytes) {
  for (size_t i = 0; i < nbytes; i++)
    dst[i] = src[i];
  return dst + nbytes;
}

/*! @abstract Emit (L,0,0) instructions (final literal).
 * We read at most \p L bytes from \p p.
 * @param p input stream
 * @param q1 the first byte after the output buffer.
 * @return pointer to the next output, <= \p q1.
 * @return \p q1 if output is full. In that case, output will be partially invalid.
 */
static inline unsigned char *emit_literal(const unsigned char *p,
                                          unsigned char *q, unsigned char *q1,
                                          size_t L) {
  size_t x;
  while (L > 15) {
    x = L < 271 ? L : 271;
    if (q + x + 10 >= q1)
      goto OUT_FULL;
    store2(q, 0xE0 + ((x - 16) << 8));
    q += 2;
    L -= x;
    q = lzvn_copy8(q, p, x);
    p += x;
  }
  if (L > 0) {
    if (q + L + 10 >= q1)
      goto OUT_FULL;
    *q++ = 0xE0 + L; // 1110LLLL
    q = lzvn_copy8(q, p, L);
  }
  return q;

OUT_FULL:
  return q1;
}

/*! @abstract Emit (L,M,D) instructions. M>=3.
 * @param p input stream pointing to the beginning of the literal. We read at
 * most \p L+4 bytes from \p p.
 * @param q1 the first byte after the output buffer.
 * @return pointer to the next output, <= \p q1.
 * @return \p q1 if output is full. In that case, output will be partially invalid.
 */
static inline unsigned char *emit(const unsigned char *p, unsigned char *q,
                                  unsigned char *q1, size_t L, size_t M,
                                  size_t D, size_t D_prev) {
  size_t x;
  while (L > 15) {
    x = L < 271 ? L : 271;
    if (q + x + 10 >= q1)
      goto OUT_FULL;
    store2(q, 0xE0 + ((x - 16) << 8));
    q += 2;
    L -= x;
    q = lzvn_copy64(q, p, x);
    p += x;
  }
  if (L > 3) {
    if (q + L + 10 >= q1)
      goto OUT_FULL;
    *q++ = 0xE0 + L; // 1110LLLL
    q = lzvn_copy64(q, p, L);
    p += L;
    L = 0;
  }
  x = M <= 10 - 2 * L ? M : 10 - 2 * L; // x = min(10-2*L,M)
  M -= x;
  x -= 3; // M = (x+3) + M'    max value for x is 7-2*L

  // Here L<4 literals remaining, we read them here
  uint32_t literal = load4(p);
  // P is not accessed after this point

  // Relaxed capacity test covering all cases
  if (q + 8 >= q1)
    goto OUT_FULL;

  if (D == D_prev) {
    if (L == 0) {
      *q++ = 0xF0 + (x + 3); // XM!
    } else {
      *q++ = (L << 6) + (x << 3) + 6; //  LLxxx110
    }
    store4(q, literal);
    q += L;
  } else if (D < 2048 - 2 * 256) {
    // Short dist    D>>8 in 0..5
    *q++ = (D >> 8) + (L << 6) + (x << 3); // LLxxxDDD
    *q++ = D & 0xFF;
    store4(q, literal);
    q += L;
  } else if (D >= (1 << 14) || M == 0 || (x + 3) + M > 34) {
    // Long dist
    *q++ = (L << 6) + (x << 3) + 7;
    store2(q, D);
    q += 2;
    store4(q, literal);
    q += L;
  } else {
    // Medium distance
    x += M;
    M = 0;
    *q++ = 0xA0 + (x >> 2) + (L << 3);
    store2(q, D << 2 | (x & 3));
    q += 2;
    store4(q, literal);
    q += L;
  }

  // Issue remaining match
  while (M > 15) {
    if (q + 2 >= q1)
      goto OUT_FULL;
    x = M < 271 ? M : 271;
    store2(q, 0xf0 + ((x - 16) << 8));
    q += 2;
    M -= x;
  }
  if (M > 0) {
    if (q + 1 >= q1)
      goto OUT_FULL;
    *q++ = 0xF0 + M; // M = 0..15
  }

  return q;

OUT_FULL:
  return q1;
}

// ===============================================================
// Conversions

/*! @abstract Return 32-bit value to store for offset x. */
static inline int32_t offset_to_s32(lzvn_offset x) { return (int32_t)x; }

/*! @abstract Get offset from 32-bit stored value x. */
static inline lzvn_offset offset_from_s32(int32_t x) { return (lzvn_offset)x; }

// ===============================================================
// Hash and Matching

/*! @abstract Get hash in range \c [0,LZVN_ENCODE_HASH_VALUES-1] from 3 bytes in i. */
static inline uint32_t hash3i(uint32_t i) {
  i &= 0xffffff; // truncate to 24-bit input (slightly increases compression ratio)
  uint32_t h = (i * (1 + (1 << 6) + (1 << 12))) >> 12;
  return h & (LZVN_ENCODE_HASH_VALUES - 1);
}

/*! @abstract Return the number [0, 4] of zero bytes in \p x, starting from the
 * least significant byte. */
static inline lzvn_offset trailing_zero_bytes(uint32_t x) {
  return (x == 0) ? 4 : (__builtin_ctzl(x) >> 3);
}

/*! @abstract Return the number [0, 4] of matching chars between values at
 * \p src+i and \p src+j, starting from the least significant byte.
 * Assumes we can read 4 chars from each position. */
static inline lzvn_offset nmatch4(const unsigned char *src, lzvn_offset i,
                                  lzvn_offset j) {
  uint32_t vi = load4(src + i);
  uint32_t vj = load4(src + j);
  return trailing_zero_bytes(vi ^ vj);
}

/*! @abstract Check if l_begin, m_begin, m0_begin (m0_begin < m_begin) can be
 * expanded to a match of length at least 3.
 * @param m_begin new string to match.
 * @param m0_begin candidate old string.
 * @param src source buffer, with valid indices src_begin <= i < src_end.
 * (src_begin may be <0)
 * @return If a match can be found, return 1 and set all \p match fields,
 * otherwise return 0.
 * @note \p *match should be 0 before the call. */
static inline int lzvn_find_match(const unsigned char *src,
                                  lzvn_offset src_begin,
                                  lzvn_offset src_end, lzvn_offset l_begin,
                                  lzvn_offset m0_begin, lzvn_offset m_begin,
                                  lzvn_match_info *match) {
  lzvn_offset n = nmatch4(src, m_begin, m0_begin);
  if (n < 3)
    return 0; // no match

  lzvn_offset D = m_begin - m0_begin; // actual distance
  if (D <= 0 || D > LZVN_ENCODE_MAX_DISTANCE)
    return 0; // distance out of range

  // Expand forward
  lzvn_offset m_end = m_begin + n;
  while (n == 4 && m_end + 4 < src_end) {
    n = nmatch4(src, m_end, m_end - D);
    m_end += n;
  }

  // Expand backwards over literal
  while (m0_begin > src_begin && m_begin > l_begin &&
         src[m_begin - 1] == src[m0_begin - 1]) {
    m0_begin--;
    m_begin--;
  }

  // OK, we keep it, update MATCH
  lzvn_offset M = m_end - m_begin; // match length
  match->m_begin = m_begin;
  match->m_end = m_end;
  match->K = M - ((D < 0x600) ? 2 : 3);
  match->M = M;
  match->D = D;

  return 1; // OK
}

/*! @abstract Same as lzvn_find_match, but we already know that N bytes do
 *  match (N<=4). */
static inline int lzvn_find_matchN(const unsigned char *src,
                                   lzvn_offset src_begin,
                                   lzvn_offset src_end, lzvn_offset l_begin,
                                   lzvn_offset m0_begin, lzvn_offset m_begin,
                                   lzvn_offset n, lzvn_match_info *match) {
  // We can skip the first comparison on 4 bytes
  if (n < 3)
    return 0; // no match

  lzvn_offset D = m_begin - m0_begin; // actual distance
  if (D <= 0 || D > LZVN_ENCODE_MAX_DISTANCE)
    return 0; // distance out of range

  // Expand forward
  lzvn_offset m_end = m_begin + n;
  while (n == 4 && m_end + 4 < src_end) {
    n = nmatch4(src, m_end, m_end - D);
    m_end += n;
  }

  // Expand backwards over literal
  while (m0_begin > src_begin && m_begin > l_begin &&
         src[m_begin - 1] == src[m0_begin - 1]) {
    m0_begin--;
    m_begin--;
  }

  // OK, we keep it, update MATCH
  lzvn_offset M = m_end - m_begin; // match length
  match->m_begin = m_begin;
  match->m_end = m_end;
  match->K = M - ((D < 0x600) ? 2 : 3);
  match->M = M;
  match->D = D;

  return 1; // OK
}

// ===============================================================
// Encoder Backend

/*! @abstract Emit a match and update state.
 * @return number of bytes written to \p dst. May be 0 if there is no more space
 * in \p dst to emit the match. */
static inline lzvn_offset lzvn_emit_match(lzvn_encoder_state *state,
                                          lzvn_match_info match) {
  size_t L = (size_t)(match.m_begin - state->src_literal); // literal count
  size_t M = (size_t)match.M;                              // match length
  size_t D = (size_t)match.D;                              // match distance
  size_t D_prev = (size_t)state->d_prev; // previously emitted match distance
  unsigned char *dst = emit(state->src + state->src_literal, state->dst,
                            state->dst_end, L, M, D, D_prev);
  // Check if DST is full
  if (dst >= state->dst_end) {
    return 0; // FULL
  }

  // Update state
  lzvn_offset dst_used = dst - state->dst;
  state->d_prev = match.D;
  state->dst = dst;
  state->src_literal = match.m_end;
  return dst_used;
}

/*! @abstract Emit a n-bytes literal and update state.
 * @return number of bytes written to \p dst. May be 0 if there is no more space
 * in \p dst to emit the literal. */
static inline lzvn_offset lzvn_emit_literal(lzvn_encoder_state *state,
                                            lzvn_offset n) {
  size_t L = (size_t)n;
  unsigned char *dst = emit_literal(state->src + state->src_literal, state->dst,
                                    state->dst_end, L);
  // Check if DST is full
  if (dst >= state->dst_end)
    return 0; // FULL

  // Update state
  lzvn_offset dst_used = dst - state->dst;
  state->dst = dst;
  state->src_literal += n;
  return dst_used;
}

/*! @abstract Emit end-of-stream and update state.
 * @return number of bytes written to \p dst. May be 0 if there is no more space
 * in \p dst to emit the instruction. */
static inline lzvn_offset lzvn_emit_end_of_stream(lzvn_encoder_state *state) {
  // Do we have 8 byte in dst?
  if (state->dst_end < state->dst + 8)
    return 0; // FULL

  // Insert end marker and update state
  store8(state->dst, 0x06); // end-of-stream command
  state->dst += 8;
  return 8; // dst_used
}

// ===============================================================
// Encoder Functions

/*! @abstract Initialize encoder table in \p state, uses current I/O parameters. */
static inline void lzvn_init_table(lzvn_encoder_state *state) {
  lzvn_offset index = -LZVN_ENCODE_MAX_DISTANCE; // max match distance
  if (index < state->src_begin)
    index = state->src_begin;
  uint32_t value = load4(state->src + index);

  lzvn_encode_entry_type e;
  for (int i = 0; i < 4; i++) {
    e.indices[i] = offset_to_s32(index);
    e.values[i] = value;
  }
  for (int u = 0; u < LZVN_ENCODE_HASH_VALUES; u++)
    state->table[u] = e; // fill entire table
}

void lzvn_encode(lzvn_encoder_state *state) {
  const lzvn_match_info NO_MATCH = {0};

  for (; state->src_current < state->src_current_end; state->src_current++) {
    // Get 4 bytes at src_current
    uint32_t vi = load4(state->src + state->src_current);

    // Compute new hash H at position I, and push value into position table
    int h = hash3i(vi); // index of first entry

    // Read table entries for H
    lzvn_encode_entry_type e = state->table[h];

    // Update entry with index=current and value=vi
    lzvn_encode_entry_type updated_e; // rotate values, so we will replace the oldest
    updated_e.indices[0] = offset_to_s32(state->src_current);
    updated_e.indices[1] = e.indices[0];
    updated_e.indices[2] = e.indices[1];
    updated_e.indices[3] = e.indices[2];
    updated_e.values[0] = vi;
    updated_e.values[1] = e.values[0];
    updated_e.values[2] = e.values[1];
    updated_e.values[3] = e.values[2];

    // Do not check matches if still in previously emitted match
    if (state->src_current < state->src_literal)
      goto after_emit;

// Update best with candidate if better
#define UPDATE(best, candidate)                                                \
  do {                                                                         \
    if (candidate.K > best.K ||                                                \
        ((candidate.K == best.K) && (candidate.m_end > best.m_end + 1))) {     \
      best = candidate;                                                        \
    }                                                                          \
  } while (0)
// Check candidate. Keep if better.
#define CHECK_CANDIDATE(ik, nk)                                                \
  do {                                                                         \
    lzvn_match_info m1;                                                              \
    if (lzvn_find_matchN(state->src, state->src_begin, state->src_end,               \
                   state->src_literal, ik, state->src_current, nk, &m1)) {     \
      UPDATE(incoming, m1);                                                    \
    }                                                                          \
  } while (0)
// Emit match M. Return if we don't have enough space in the destination buffer
#define EMIT_MATCH(m)                                                          \
  do {                                                                         \
    if (lzvn_emit_match(state, m) == 0)                                              \
      return;                                                                  \
  } while (0)
// Emit literal of length L. Return if we don't have enough space in the
// destination buffer
#define EMIT_LITERAL(l)                                                        \
  do {                                                                         \
    if (lzvn_emit_literal(state, l) == 0)                                            \
      return;                                                                  \
  } while (0)

    lzvn_match_info incoming = NO_MATCH;

    // Check candidates in order (closest first)
    uint32_t diffs[4];
    for (int k = 0; k < 4; k++)
      diffs[k] = e.values[k] ^ vi; // XOR, 0 if equal
    lzvn_offset ik;                // index
    lzvn_offset nk;                // match byte count

    // The values stored in e.xyzw are 32-bit signed indices, extended to signed
    // type lzvn_offset
    ik = offset_from_s32(e.indices[0]);
    nk = trailing_zero_bytes(diffs[0]);
    CHECK_CANDIDATE(ik, nk);
    ik = offset_from_s32(e.indices[1]);
    nk = trailing_zero_bytes(diffs[1]);
    CHECK_CANDIDATE(ik, nk);
    ik = offset_from_s32(e.indices[2]);
    nk = trailing_zero_bytes(diffs[2]);
    CHECK_CANDIDATE(ik, nk);
    ik = offset_from_s32(e.indices[3]);
    nk = trailing_zero_bytes(diffs[3]);
    CHECK_CANDIDATE(ik, nk);

    // Check candidate at previous distance
    if (state->d_prev != 0) {
      lzvn_match_info m1;
      if (lzvn_find_match(state->src, state->src_begin, state->src_end,
                    state->src_literal, state->src_current - state->d_prev,
                    state->src_current, &m1)) {
        m1.K = m1.M - 1; // fix K for D_prev
        UPDATE(incoming, m1);
      }
    }

    // Here we have the best candidate in incoming, may be NO_MATCH

    // If no incoming match, and literal backlog becomes too high, emit pending
    // match, or literals if there is no pending match
    if (incoming.M == 0) {
      if (state->src_current - state->src_literal >=
          LZVN_ENCODE_MAX_LITERAL_BACKLOG) // at this point, we always have
                                           // current >= literal
      {
        if (state->pending.M != 0) {
          EMIT_MATCH(state->pending);
          state->pending = NO_MATCH;
        } else {
          EMIT_LITERAL(271); // emit long literal (271 is the longest literal size we allow)
        }
      }
      goto after_emit;
    }

    if (state->pending.M == 0) {
      // NOTE. Here, we can also emit incoming right away. It will make the
      // encoder 1.5x faster, at a cost of ~10% lower compression ratio:
      // EMIT_MATCH(incoming);
      // state->pending = NO_MATCH;

      // No pending match, emit nothing, keep incoming
      state->pending = incoming;
    } else {
      // Here we have both incoming and pending
      if (state->pending.m_end <= incoming.m_begin) {
        // No overlap: emit pending, keep incoming
        EMIT_MATCH(state->pending);
        state->pending = incoming;
      } else {
        // If pending is better, emit pending and discard incoming.
        // Otherwise, emit incoming and discard pending.
        if (incoming.K > state->pending.K)
          state->pending = incoming;
        EMIT_MATCH(state->pending);
        state->pending = NO_MATCH;
      }
    }

  after_emit:

    // We commit state changes only after we tried to emit instructions, so we
    // can restart in the same state in case dst was full and we quit the loop.
    state->table[h] = updated_e;

  } // i loop

  // Do not emit pending match here. We do it only at the end of stream.
}

// ===============================================================
// API entry points

size_t lzvn_encode_scratch_size(void) { return LZVN_ENCODE_WORK_SIZE; }

static size_t lzvn_encode_partial(void *__restrict dst, size_t dst_size,
                                  const void *__restrict src, size_t src_size,
                                  size_t *src_used, void *__restrict work) {
  // Min size checks to avoid accessing memory outside buffers.
  if (dst_size < LZVN_ENCODE_MIN_DST_SIZE) {
    *src_used = 0;
    return 0;
  }
  // Max input size check (limit to offsets on uint32_t).
  if (src_size > LZVN_ENCODE_MAX_SRC_SIZE) {
    src_size = LZVN_ENCODE_MAX_SRC_SIZE;
  }

  // Setup encoder state
  lzvn_encoder_state state;
  memset(&state, 0, sizeof(state));

  state.src = src;
  state.src_begin = 0;
  state.src_end = (lzvn_offset)src_size;
  state.src_literal = 0;
  state.src_current = 0;
  state.dst = dst;
  state.dst_begin = dst;
  state.dst_end = (unsigned char *)dst + dst_size - 8; // reserve 8 bytes for end-of-stream
  state.table = work;

  // Do not encode if the input buffer is too small. We'll emit a literal instead.
  if (src_size >= LZVN_ENCODE_MIN_SRC_SIZE) {

    state.src_current_end = (lzvn_offset)src_size - LZVN_ENCODE_MIN_MARGIN;
    lzvn_init_table(&state);
    lzvn_encode(&state);

  }

  // No need to test the return value: src_literal will not be updated on failure,
  // and we will fail later.
  lzvn_emit_literal(&state, state.src_end - state.src_literal);

  // Restore original size, so end-of-stream always succeeds, and emit it
  state.dst_end = (unsigned char *)dst + dst_size;
  lzvn_emit_end_of_stream(&state);

  *src_used = state.src_literal;
  return (size_t)(state.dst - state.dst_begin);
}

size_t lzvn_encode_buffer(void *__restrict dst, size_t dst_size,
                          const void *__restrict src, size_t src_size,
                          void *__restrict work) {
  size_t src_used = 0;
  size_t dst_used =
      lzvn_encode_partial(dst, dst_size, src, src_size, &src_used, work);
  if (src_used != src_size)
    return 0;      // could not encode entire input stream = fail
  return dst_used; // return encoded size
}

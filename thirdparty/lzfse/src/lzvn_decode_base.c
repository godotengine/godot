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

// LZVN low-level decoder

#include "lzvn_decode_base.h"

#if !defined(HAVE_LABELS_AS_VALUES)
#  if defined(__GNUC__) || defined(__clang__)
#    define HAVE_LABELS_AS_VALUES 1
#  else
#    define HAVE_LABELS_AS_VALUES 0
#  endif
#endif

//  Both the source and destination buffers are represented by a pointer and
//  a length; they are *always* updated in concert using this macro; however
//  many bytes the pointer is advanced, the length is decremented by the same
//  amount. Thus, pointer + length always points to the byte one past the end
//  of the buffer.
#define PTR_LEN_INC(_pointer, _length, _increment)                             \
  (_pointer += _increment, _length -= _increment)

//  Update state with current positions and distance, corresponding to the
//  beginning of an instruction in both streams
#define UPDATE_GOOD                                                            \
  (state->src = src_ptr, state->dst = dst_ptr, state->d_prev = D)

void lzvn_decode(lzvn_decoder_state *state) {
#if HAVE_LABELS_AS_VALUES
  // Jump table for all instructions
  static const void *opc_tbl[256] = {
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&eos,   &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&nop,   &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&nop,   &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&udef,  &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&udef,  &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&udef,  &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&udef,  &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&udef,  &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,
      &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d,
      &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d,
      &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d,
      &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d, &&med_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&sml_d, &&pre_d, &&lrg_d,
      &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,
      &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,  &&udef,
      &&lrg_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l,
      &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l, &&sml_l,
      &&lrg_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m,
      &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m, &&sml_m};
#endif
  size_t src_len = state->src_end - state->src;
  size_t dst_len = state->dst_end - state->dst;
  if (src_len == 0 || dst_len == 0)
    return; // empty buffer

  const unsigned char *src_ptr = state->src;
  unsigned char *dst_ptr = state->dst;
  size_t D = state->d_prev;
  size_t M;
  size_t L;
  size_t opc_len;

  // Do we have a partially expanded match saved in state?
  if (state->L != 0 || state->M != 0) {
    L = state->L;
    M = state->M;
    D = state->D;
    opc_len = 0; // we already skipped the op
    state->L = state->M = state->D = 0;
    if (M == 0)
      goto copy_literal;
    if (L == 0)
      goto copy_match;
    goto copy_literal_and_match;
  }

  unsigned char opc = src_ptr[0];

#if HAVE_LABELS_AS_VALUES
  goto *opc_tbl[opc];
#else
  for (;;) {
    switch (opc) {
#endif
//  ===============================================================
//  These four opcodes (sml_d, med_d, lrg_d, and pre_d) encode both a
//  literal and a match. The bulk of their implementations are shared;
//  each label here only does the work of setting the opcode length (not
//  including any literal bytes), and extracting the literal length, match
//  length, and match distance (except in pre_d). They then jump into the
//  shared implementation to actually output the literal and match bytes.
//
//  No error checking happens in the first stage, except for ensuring that
//  the source has enough length to represent the full opcode before
//  reading past the first byte.
sml_d:
#if !HAVE_LABELS_AS_VALUES
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 8:
  case 9:
  case 10:
  case 11:
  case 12:
  case 13:
  case 16:
  case 17:
  case 18:
  case 19:
  case 20:
  case 21:
  case 24:
  case 25:
  case 26:
  case 27:
  case 28:
  case 29:
  case 32:
  case 33:
  case 34:
  case 35:
  case 36:
  case 37:
  case 40:
  case 41:
  case 42:
  case 43:
  case 44:
  case 45:
  case 48:
  case 49:
  case 50:
  case 51:
  case 52:
  case 53:
  case 56:
  case 57:
  case 58:
  case 59:
  case 60:
  case 61:
  case 64:
  case 65:
  case 66:
  case 67:
  case 68:
  case 69:
  case 72:
  case 73:
  case 74:
  case 75:
  case 76:
  case 77:
  case 80:
  case 81:
  case 82:
  case 83:
  case 84:
  case 85:
  case 88:
  case 89:
  case 90:
  case 91:
  case 92:
  case 93:
  case 96:
  case 97:
  case 98:
  case 99:
  case 100:
  case 101:
  case 104:
  case 105:
  case 106:
  case 107:
  case 108:
  case 109:
  case 128:
  case 129:
  case 130:
  case 131:
  case 132:
  case 133:
  case 136:
  case 137:
  case 138:
  case 139:
  case 140:
  case 141:
  case 144:
  case 145:
  case 146:
  case 147:
  case 148:
  case 149:
  case 152:
  case 153:
  case 154:
  case 155:
  case 156:
  case 157:
  case 192:
  case 193:
  case 194:
  case 195:
  case 196:
  case 197:
  case 200:
  case 201:
  case 202:
  case 203:
  case 204:
  case 205:
#endif
  UPDATE_GOOD;
  // "small distance": This opcode has the structure LLMMMDDD DDDDDDDD LITERAL
  //  where the length of literal (0-3 bytes) is encoded by the high 2 bits of
  //  the first byte. We first extract the literal length so we know how long
  //  the opcode is, then check that the source can hold both this opcode and
  //  at least one byte of the next (because any valid input stream must be
  //  terminated with an eos token).
  opc_len = 2;
  L = (size_t)extract(opc, 6, 2);
  M = (size_t)extract(opc, 3, 3) + 3;
  //  We need to ensure that the source buffer is long enough that we can
  //  safely read this entire opcode, the literal that follows, and the first
  //  byte of the next opcode.  Once we satisfy this requirement, we can
  //  safely unpack the match distance. A check similar to this one is
  //  present in all the opcode implementations.
  if (src_len <= opc_len + L)
    return; // source truncated
  D = (size_t)extract(opc, 0, 3) << 8 | src_ptr[1];
  goto copy_literal_and_match;

med_d:
#if !HAVE_LABELS_AS_VALUES
  case 160:
  case 161:
  case 162:
  case 163:
  case 164:
  case 165:
  case 166:
  case 167:
  case 168:
  case 169:
  case 170:
  case 171:
  case 172:
  case 173:
  case 174:
  case 175:
  case 176:
  case 177:
  case 178:
  case 179:
  case 180:
  case 181:
  case 182:
  case 183:
  case 184:
  case 185:
  case 186:
  case 187:
  case 188:
  case 189:
  case 190:
  case 191:
#endif
  UPDATE_GOOD;
  //  "medium distance": This is a minor variant of the "small distance"
  //  encoding, where we will now use two extra bytes instead of one to encode
  //  the restof the match length and distance. This allows an extra two bits
  //  for the match length, and an extra three bits for the match distance. The
  //  full structure of the opcode is 101LLMMM DDDDDDMM DDDDDDDD LITERAL.
  opc_len = 3;
  L = (size_t)extract(opc, 3, 2);
  if (src_len <= opc_len + L)
    return; // source truncated
  uint16_t opc23 = load2(&src_ptr[1]);
  M = (size_t)((extract(opc, 0, 3) << 2 | extract(opc23, 0, 2)) + 3);
  D = (size_t)extract(opc23, 2, 14);
  goto copy_literal_and_match;

lrg_d:
#if !HAVE_LABELS_AS_VALUES
  case 7:
  case 15:
  case 23:
  case 31:
  case 39:
  case 47:
  case 55:
  case 63:
  case 71:
  case 79:
  case 87:
  case 95:
  case 103:
  case 111:
  case 135:
  case 143:
  case 151:
  case 159:
  case 199:
  case 207:
#endif
  UPDATE_GOOD;
  //  "large distance": This is another variant of the "small distance"
  //  encoding, where we will now use two extra bytes to encode the match
  //  distance, which allows distances up to 65535 to be represented. The full
  //  structure of the opcode is LLMMM111 DDDDDDDD DDDDDDDD LITERAL.
  opc_len = 3;
  L = (size_t)extract(opc, 6, 2);
  M = (size_t)extract(opc, 3, 3) + 3;
  if (src_len <= opc_len + L)
    return; // source truncated
  D = load2(&src_ptr[1]);
  goto copy_literal_and_match;

pre_d:
#if !HAVE_LABELS_AS_VALUES
  case 70:
  case 78:
  case 86:
  case 94:
  case 102:
  case 110:
  case 134:
  case 142:
  case 150:
  case 158:
  case 198:
  case 206:
#endif
  UPDATE_GOOD;
  //  "previous distance": This opcode has the structure LLMMM110, where the
  //  length of the literal (0-3 bytes) is encoded by the high 2 bits of the
  //  first byte. We first extract the literal length so we know how long
  //  the opcode is, then check that the source can hold both this opcode and
  //  at least one byte of the next (because any valid input stream must be
  //  terminated with an eos token).
  opc_len = 1;
  L = (size_t)extract(opc, 6, 2);
  M = (size_t)extract(opc, 3, 3) + 3;
  if (src_len <= opc_len + L)
    return; // source truncated
  goto copy_literal_and_match;

copy_literal_and_match:
  //  Common implementation of writing data for opcodes that have both a
  //  literal and a match. We begin by advancing the source pointer past
  //  the opcode, so that it points at the first literal byte (if L
  //  is non-zero; otherwise it points at the next opcode).
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  //  Now we copy the literal from the source pointer to the destination.
  if (__builtin_expect(dst_len >= 4 && src_len >= 4, 1)) {
    //  The literal is 0-3 bytes; if we are not near the end of the buffer,
    //  we can safely just do a 4 byte copy (which is guaranteed to cover
    //  the complete literal, and may include some other bytes as well).
    store4(dst_ptr, load4(src_ptr));
  } else if (L <= dst_len) {
    //  We are too close to the end of either the input or output stream
    //  to be able to safely use a four-byte copy, but we will not exhaust
    //  either stream (we already know that the source will not be
    //  exhausted from checks in the individual opcode implementations,
    //  and we just tested that dst_len > L). Thus, we need to do a
    //  byte-by-byte copy of the literal. This is slow, but it can only ever
    //  happen near the very end of a buffer, so it is not an important case to
    //  optimize.
    for (size_t i = 0; i < L; ++i)
      dst_ptr[i] = src_ptr[i];
  } else {
    // Destination truncated: fill DST, and store partial match

    // Copy partial literal
    for (size_t i = 0; i < dst_len; ++i)
      dst_ptr[i] = src_ptr[i];
    // Save state
    state->src = src_ptr + dst_len;
    state->dst = dst_ptr + dst_len;
    state->L = L - dst_len;
    state->M = M;
    state->D = D;
    return; // destination truncated
  }
  //  Having completed the copy of the literal, we advance both the source
  //  and destination pointers by the number of literal bytes.
  PTR_LEN_INC(dst_ptr, dst_len, L);
  PTR_LEN_INC(src_ptr, src_len, L);
  //  Check if the match distance is valid; matches must not reference
  //  bytes that preceed the start of the output buffer, nor can the match
  //  distance be zero.
  if (D > dst_ptr - state->dst_begin || D == 0)
    goto invalid_match_distance;
copy_match:
  //  Now we copy the match from dst_ptr - D to dst_ptr. It is important to keep
  //  in mind that we may have D < M, in which case the source and destination
  //  windows overlap in the copy. The semantics of the match copy are *not*
  //  those of memmove( ); if the buffers overlap it needs to behave as though
  //  we were copying byte-by-byte in increasing address order. If, for example,
  //  D is 1, the copy operation is equivalent to:
  //
  //      memset(dst_ptr, dst_ptr[-1], M);
  //
  //  i.e. it splats the previous byte. This means that we need to be very
  //  careful about using wide loads or stores to perform the copy operation.
  if (__builtin_expect(dst_len >= M + 7 && D >= 8, 1)) {
    //  We are not near the end of the buffer, and the match distance
    //  is at least eight. Thus, we can safely loop using eight byte
    //  copies. The last of these may slop over the intended end of
    //  the match, but this is OK because we know we have a safety bound
    //  away from the end of the destination buffer.
    for (size_t i = 0; i < M; i += 8)
      store8(&dst_ptr[i], load8(&dst_ptr[i - D]));
  } else if (M <= dst_len) {
    //  Either the match distance is too small, or we are too close to
    //  the end of the buffer to safely use eight byte copies. Fall back
    //  on a simple byte-by-byte implementation.
    for (size_t i = 0; i < M; ++i)
      dst_ptr[i] = dst_ptr[i - D];
  } else {
    // Destination truncated: fill DST, and store partial match

    // Copy partial match
    for (size_t i = 0; i < dst_len; ++i)
      dst_ptr[i] = dst_ptr[i - D];
    // Save state
    state->src = src_ptr;
    state->dst = dst_ptr + dst_len;
    state->L = 0;
    state->M = M - dst_len;
    state->D = D;
    return; // destination truncated
  }
  //  Update the destination pointer and length to account for the bytes
  //  written by the match, then load the next opcode byte and branch to
  //  the appropriate implementation.
  PTR_LEN_INC(dst_ptr, dst_len, M);
  opc = src_ptr[0];
#if HAVE_LABELS_AS_VALUES
  goto *opc_tbl[opc];
#else
  break;
#endif

// ===============================================================
// Opcodes representing only a match (no literal).
//  These two opcodes (lrg_m and sml_m) encode only a match. The match
//  distance is carried over from the previous opcode, so all they need
//  to encode is the match length. We are able to reuse the match copy
//  sequence from the literal and match opcodes to perform the actual
//  copy implementation.
sml_m:
#if !HAVE_LABELS_AS_VALUES
  case 241:
  case 242:
  case 243:
  case 244:
  case 245:
  case 246:
  case 247:
  case 248:
  case 249:
  case 250:
  case 251:
  case 252:
  case 253:
  case 254:
  case 255:
#endif
  UPDATE_GOOD;
  //  "small match": This opcode has no literal, and uses the previous match
  //  distance (i.e. it encodes only the match length), in a single byte as
  //  1111MMMM.
  opc_len = 1;
  if (src_len <= opc_len)
    return; // source truncated
  M = (size_t)extract(opc, 0, 4);
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  goto copy_match;

lrg_m:
#if !HAVE_LABELS_AS_VALUES
  case 240:
#endif
  UPDATE_GOOD;
  //  "large match": This opcode has no literal, and uses the previous match
  //  distance (i.e. it encodes only the match length). It is encoded in two
  //  bytes as 11110000 MMMMMMMM.  Because matches smaller than 16 bytes can
  //  be represented by sml_m, there is an implicit bias of 16 on the match
  //  length; the representable values are [16,271].
  opc_len = 2;
  if (src_len <= opc_len)
    return; // source truncated
  M = src_ptr[1] + 16;
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  goto copy_match;

// ===============================================================
// Opcodes representing only a literal (no match).
//  These two opcodes (lrg_l and sml_l) encode only a literal.  There is no
//  match length or match distance to worry about (but we need to *not*
//  touch D, as it must be preserved between opcodes).
sml_l:
#if !HAVE_LABELS_AS_VALUES
  case 225:
  case 226:
  case 227:
  case 228:
  case 229:
  case 230:
  case 231:
  case 232:
  case 233:
  case 234:
  case 235:
  case 236:
  case 237:
  case 238:
  case 239:
#endif
  UPDATE_GOOD;
  //  "small literal": This opcode has no match, and encodes only a literal
  //  of length up to 15 bytes. The format is 1110LLLL LITERAL.
  opc_len = 1;
  L = (size_t)extract(opc, 0, 4);
  goto copy_literal;

lrg_l:
#if !HAVE_LABELS_AS_VALUES
  case 224:
#endif
  UPDATE_GOOD;
  //  "large literal": This opcode has no match, and uses the previous match
  //  distance (i.e. it encodes only the match length). It is encoded in two
  //  bytes as 11100000 LLLLLLLL LITERAL.  Because literals smaller than 16
  //  bytes can be represented by sml_l, there is an implicit bias of 16 on
  //  the literal length; the representable values are [16,271].
  opc_len = 2;
  if (src_len <= 2)
    return; // source truncated
  L = src_ptr[1] + 16;
  goto copy_literal;

copy_literal:
  //  Check that the source buffer is large enough to hold the complete
  //  literal and at least the first byte of the next opcode. If so, advance
  //  the source pointer to point to the first byte of the literal and adjust
  //  the source length accordingly.
  if (src_len <= opc_len + L)
    return; // source truncated
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  //  Now we copy the literal from the source pointer to the destination.
  if (dst_len >= L + 7 && src_len >= L + 7) {
    //  We are not near the end of the source or destination buffers; thus
    //  we can safely copy the literal using wide copies, without worrying
    //  about reading or writing past the end of either buffer.
    for (size_t i = 0; i < L; i += 8)
      store8(&dst_ptr[i], load8(&src_ptr[i]));
  } else if (L <= dst_len) {
    //  We are too close to the end of either the input or output stream
    //  to be able to safely use an eight-byte copy. Instead we copy the
    //  literal byte-by-byte.
    for (size_t i = 0; i < L; ++i)
      dst_ptr[i] = src_ptr[i];
  } else {
    // Destination truncated: fill DST, and store partial match

    // Copy partial literal
    for (size_t i = 0; i < dst_len; ++i)
      dst_ptr[i] = src_ptr[i];
    // Save state
    state->src = src_ptr + dst_len;
    state->dst = dst_ptr + dst_len;
    state->L = L - dst_len;
    state->M = 0;
    state->D = D;
    return; // destination truncated
  }
  //  Having completed the copy of the literal, we advance both the source
  //  and destination pointers by the number of literal bytes.
  PTR_LEN_INC(dst_ptr, dst_len, L);
  PTR_LEN_INC(src_ptr, src_len, L);
  //  Load the first byte of the next opcode, and jump to its implementation.
  opc = src_ptr[0];
#if HAVE_LABELS_AS_VALUES
  goto *opc_tbl[opc];
#else
  break;
#endif

// ===============================================================
// Other opcodes
nop:
#if !HAVE_LABELS_AS_VALUES
  case 14:
  case 22:
#endif
  UPDATE_GOOD;
  opc_len = 1;
  if (src_len <= opc_len)
    return; // source truncated
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  opc = src_ptr[0];
#if HAVE_LABELS_AS_VALUES
  goto *opc_tbl[opc];
#else
  break;
#endif

eos:
#if !HAVE_LABELS_AS_VALUES
  case 6:
#endif
  opc_len = 8;
  if (src_len < opc_len)
    return; // source truncated (here we don't need an extra byte for next op
            // code)
  PTR_LEN_INC(src_ptr, src_len, opc_len);
  state->end_of_stream = 1;
  UPDATE_GOOD;
  return; // end-of-stream

// ===============================================================
// Return on error
udef:
#if !HAVE_LABELS_AS_VALUES
  case 30:
  case 38:
  case 46:
  case 54:
  case 62:
  case 112:
  case 113:
  case 114:
  case 115:
  case 116:
  case 117:
  case 118:
  case 119:
  case 120:
  case 121:
  case 122:
  case 123:
  case 124:
  case 125:
  case 126:
  case 127:
  case 208:
  case 209:
  case 210:
  case 211:
  case 212:
  case 213:
  case 214:
  case 215:
  case 216:
  case 217:
  case 218:
  case 219:
  case 220:
  case 221:
  case 222:
  case 223:
#endif
invalid_match_distance:

  return; // we already updated state
#if !HAVE_LABELS_AS_VALUES
    }
  }
#endif
}

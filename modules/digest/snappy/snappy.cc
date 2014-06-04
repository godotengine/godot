// Copyright 2005 Google Inc. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "snappy.h"
#include "snappy-internal.h"
#include "snappy-sinksource.h"

#include <stdio.h>

#include <algorithm>
#include <string>
#include <vector>


namespace snappy {

// Any hash function will produce a valid compressed bitstream, but a good
// hash function reduces the number of collisions and thus yields better
// compression for compressible input, and more speed for incompressible
// input. Of course, it doesn't hurt if the hash function is reasonably fast
// either, as it gets called a lot.
static inline uint32 HashBytes(uint32 bytes, int shift) {
  uint32 kMul = 0x1e35a7bd;
  return (bytes * kMul) >> shift;
}
static inline uint32 Hash(const char* p, int shift) {
  return HashBytes(UNALIGNED_LOAD32(p), shift);
}

size_t MaxCompressedLength(size_t source_len) {
  // Compressed data can be defined as:
  //    compressed := item* literal*
  //    item       := literal* copy
  //
  // The trailing literal sequence has a space blowup of at most 62/60
  // since a literal of length 60 needs one tag byte + one extra byte
  // for length information.
  //
  // Item blowup is trickier to measure.  Suppose the "copy" op copies
  // 4 bytes of data.  Because of a special check in the encoding code,
  // we produce a 4-byte copy only if the offset is < 65536.  Therefore
  // the copy op takes 3 bytes to encode, and this type of item leads
  // to at most the 62/60 blowup for representing literals.
  //
  // Suppose the "copy" op copies 5 bytes of data.  If the offset is big
  // enough, it will take 5 bytes to encode the copy op.  Therefore the
  // worst case here is a one-byte literal followed by a five-byte copy.
  // I.e., 6 bytes of input turn into 7 bytes of "compressed" data.
  //
  // This last factor dominates the blowup, so the final estimate is:
  return 32 + source_len + source_len/6;
}

enum {
  LITERAL = 0,
  COPY_1_BYTE_OFFSET = 1,  // 3 bit length + 3 bits of offset in opcode
  COPY_2_BYTE_OFFSET = 2,
  COPY_4_BYTE_OFFSET = 3
};
static const int kMaximumTagLength = 5;  // COPY_4_BYTE_OFFSET plus the actual offset.

// Copy "len" bytes from "src" to "op", one byte at a time.  Used for
// handling COPY operations where the input and output regions may
// overlap.  For example, suppose:
//    src    == "ab"
//    op     == src + 2
//    len    == 20
// After IncrementalCopy(src, op, len), the result will have
// eleven copies of "ab"
//    ababababababababababab
// Note that this does not match the semantics of either memcpy()
// or memmove().
static inline void IncrementalCopy(const char* src, char* op, ssize_t len) {
  assert(len > 0);
  do {
    *op++ = *src++;
  } while (--len > 0);
}

// Equivalent to IncrementalCopy except that it can write up to ten extra
// bytes after the end of the copy, and that it is faster.
//
// The main part of this loop is a simple copy of eight bytes at a time until
// we've copied (at least) the requested amount of bytes.  However, if op and
// src are less than eight bytes apart (indicating a repeating pattern of
// length < 8), we first need to expand the pattern in order to get the correct
// results. For instance, if the buffer looks like this, with the eight-byte
// <src> and <op> patterns marked as intervals:
//
//    abxxxxxxxxxxxx
//    [------]           src
//      [------]         op
//
// a single eight-byte copy from <src> to <op> will repeat the pattern once,
// after which we can move <op> two bytes without moving <src>:
//
//    ababxxxxxxxxxx
//    [------]           src
//        [------]       op
//
// and repeat the exercise until the two no longer overlap.
//
// This allows us to do very well in the special case of one single byte
// repeated many times, without taking a big hit for more general cases.
//
// The worst case of extra writing past the end of the match occurs when
// op - src == 1 and len == 1; the last copy will read from byte positions
// [0..7] and write to [4..11], whereas it was only supposed to write to
// position 1. Thus, ten excess bytes.

namespace {

const int kMaxIncrementCopyOverflow = 10;

inline void IncrementalCopyFastPath(const char* src, char* op, ssize_t len) {
  while (op - src < 8) {
    UnalignedCopy64(src, op);
    len -= op - src;
    op += op - src;
  }
  while (len > 0) {
    UnalignedCopy64(src, op);
    src += 8;
    op += 8;
    len -= 8;
  }
}

}  // namespace

static inline char* EmitLiteral(char* op,
                                const char* literal,
                                int len,
                                bool allow_fast_path) {
  int n = len - 1;      // Zero-length literals are disallowed
  if (n < 60) {
    // Fits in tag byte
    *op++ = LITERAL | (n << 2);

    // The vast majority of copies are below 16 bytes, for which a
    // call to memcpy is overkill. This fast path can sometimes
    // copy up to 15 bytes too much, but that is okay in the
    // main loop, since we have a bit to go on for both sides:
    //
    //   - The input will always have kInputMarginBytes = 15 extra
    //     available bytes, as long as we're in the main loop, and
    //     if not, allow_fast_path = false.
    //   - The output will always have 32 spare bytes (see
    //     MaxCompressedLength).
    if (allow_fast_path && len <= 16) {
      UnalignedCopy64(literal, op);
      UnalignedCopy64(literal + 8, op + 8);
      return op + len;
    }
  } else {
    // Encode in upcoming bytes
    char* base = op;
    int count = 0;
    op++;
    while (n > 0) {
      *op++ = n & 0xff;
      n >>= 8;
      count++;
    }
    assert(count >= 1);
    assert(count <= 4);
    *base = LITERAL | ((59+count) << 2);
  }
  memcpy(op, literal, len);
  return op + len;
}

static inline char* EmitCopyLessThan64(char* op, size_t offset, int len) {
  assert(len <= 64);
  assert(len >= 4);
  assert(offset < 65536);

  if ((len < 12) && (offset < 2048)) {
    size_t len_minus_4 = len - 4;
    assert(len_minus_4 < 8);            // Must fit in 3 bits
    *op++ = COPY_1_BYTE_OFFSET + ((len_minus_4) << 2) + ((offset >> 8) << 5);
    *op++ = offset & 0xff;
  } else {
    *op++ = COPY_2_BYTE_OFFSET + ((len-1) << 2);
    LittleEndian::Store16(op, offset);
    op += 2;
  }
  return op;
}

static inline char* EmitCopy(char* op, size_t offset, int len) {
  // Emit 64 byte copies but make sure to keep at least four bytes reserved
  while (len >= 68) {
    op = EmitCopyLessThan64(op, offset, 64);
    len -= 64;
  }

  // Emit an extra 60 byte copy if have too much data to fit in one copy
  if (len > 64) {
    op = EmitCopyLessThan64(op, offset, 60);
    len -= 60;
  }

  // Emit remainder
  op = EmitCopyLessThan64(op, offset, len);
  return op;
}


bool GetUncompressedLength(const char* start, size_t n, size_t* result) {
  uint32 v = 0;
  const char* limit = start + n;
  if (Varint::Parse32WithLimit(start, limit, &v) != NULL) {
    *result = v;
    return true;
  } else {
    return false;
  }
}

namespace internal {
uint16* WorkingMemory::GetHashTable(size_t input_size, int* table_size) {
  // Use smaller hash table when input.size() is smaller, since we
  // fill the table, incurring O(hash table size) overhead for
  // compression, and if the input is short, we won't need that
  // many hash table entries anyway.
  assert(kMaxHashTableSize >= 256);
  size_t htsize = 256;
  while (htsize < kMaxHashTableSize && htsize < input_size) {
    htsize <<= 1;
  }

  uint16* table;
  if (htsize <= ARRAYSIZE(small_table_)) {
    table = small_table_;
  } else {
    if (large_table_ == NULL) {
      large_table_ = new uint16[kMaxHashTableSize];
    }
    table = large_table_;
  }

  *table_size = htsize;
  memset(table, 0, htsize * sizeof(*table));
  return table;
}
}  // end namespace internal

// For 0 <= offset <= 4, GetUint32AtOffset(GetEightBytesAt(p), offset) will
// equal UNALIGNED_LOAD32(p + offset).  Motivation: On x86-64 hardware we have
// empirically found that overlapping loads such as
//  UNALIGNED_LOAD32(p) ... UNALIGNED_LOAD32(p+1) ... UNALIGNED_LOAD32(p+2)
// are slower than UNALIGNED_LOAD64(p) followed by shifts and casts to uint32.
//
// We have different versions for 64- and 32-bit; ideally we would avoid the
// two functions and just inline the UNALIGNED_LOAD64 call into
// GetUint32AtOffset, but GCC (at least not as of 4.6) is seemingly not clever
// enough to avoid loading the value multiple times then. For 64-bit, the load
// is done when GetEightBytesAt() is called, whereas for 32-bit, the load is
// done at GetUint32AtOffset() time.

#ifdef ARCH_K8

typedef uint64 EightBytesReference;

static inline EightBytesReference GetEightBytesAt(const char* ptr) {
  return UNALIGNED_LOAD64(ptr);
}

static inline uint32 GetUint32AtOffset(uint64 v, int offset) {
  assert(offset >= 0);
  assert(offset <= 4);
  return v >> (LittleEndian::IsLittleEndian() ? 8 * offset : 32 - 8 * offset);
}

#else

typedef const char* EightBytesReference;

static inline EightBytesReference GetEightBytesAt(const char* ptr) {
  return ptr;
}

static inline uint32 GetUint32AtOffset(const char* v, int offset) {
  assert(offset >= 0);
  assert(offset <= 4);
  return UNALIGNED_LOAD32(v + offset);
}

#endif

// Flat array compression that does not emit the "uncompressed length"
// prefix. Compresses "input" string to the "*op" buffer.
//
// REQUIRES: "input" is at most "kBlockSize" bytes long.
// REQUIRES: "op" points to an array of memory that is at least
// "MaxCompressedLength(input.size())" in size.
// REQUIRES: All elements in "table[0..table_size-1]" are initialized to zero.
// REQUIRES: "table_size" is a power of two
//
// Returns an "end" pointer into "op" buffer.
// "end - op" is the compressed size of "input".
namespace internal {
char* CompressFragment(const char* input,
                       size_t input_size,
                       char* op,
                       uint16* table,
                       const int table_size) {
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0); // table must be power of two
  const int shift = 32 - Bits::Log2Floor(table_size);
  assert(static_cast<int>(kuint32max >> shift) == table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;
  // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
  // [next_emit, ip_end) after the main loop.
  const char* next_emit = ip;

  const size_t kInputMarginBytes = 15;
  if (PREDICT_TRUE(input_size >= kInputMarginBytes)) {
    const char* ip_limit = input + input_size - kInputMarginBytes;

    for (uint32 next_hash = Hash(++ip, shift); ; ) {
      assert(next_emit < ip);
      // The body of this loop calls EmitLiteral once and then EmitCopy one or
      // more times.  (The exception is that when we're close to exhausting
      // the input we goto emit_remainder.)
      //
      // In the first iteration of this loop we're just starting, so
      // there's nothing to copy, so calling EmitLiteral once is
      // necessary.  And we only start a new iteration when the
      // current iteration has determined that a call to EmitLiteral will
      // precede the next call to EmitCopy (if any).
      //
      // Step 1: Scan forward in the input looking for a 4-byte-long match.
      // If we get close to exhausting the input then goto emit_remainder.
      //
      // Heuristic match skipping: If 32 bytes are scanned with no matches
      // found, start looking only at every other byte. If 32 more bytes are
      // scanned, look at every third byte, etc.. When a match is found,
      // immediately go back to looking at every byte. This is a small loss
      // (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      uint32 skip = 32;

      const char* next_ip = ip;
      const char* candidate;
      do {
        ip = next_ip;
        uint32 hash = next_hash;
        assert(hash == Hash(ip, shift));
        uint32 bytes_between_hash_lookups = skip++ >> 5;
        next_ip = ip + bytes_between_hash_lookups;
        if (PREDICT_FALSE(next_ip > ip_limit)) {
          goto emit_remainder;
        }
        next_hash = Hash(next_ip, shift);
        candidate = base_ip + table[hash];
        assert(candidate >= base_ip);
        assert(candidate < ip);

        table[hash] = ip - base_ip;
      } while (PREDICT_TRUE(UNALIGNED_LOAD32(ip) !=
                            UNALIGNED_LOAD32(candidate)));

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      assert(next_emit + 16 <= ip_end);
      op = EmitLiteral(op, next_emit, ip - next_emit, true);

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
      EightBytesReference input_bytes;
      uint32 candidate_bytes = 0;

      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        int matched = 4 + FindMatchLength(candidate + 4, ip + 4, ip_end);
        ip += matched;
        size_t offset = base - candidate;
        assert(0 == memcmp(base, candidate, matched));
        op = EmitCopy(op, offset, matched);
        // We could immediately start working at ip now, but to improve
        // compression we first update table[Hash(ip - 1, ...)].
        const char* insert_tail = ip - 1;
        next_emit = ip;
        if (PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        input_bytes = GetEightBytesAt(insert_tail);
        uint32 prev_hash = HashBytes(GetUint32AtOffset(input_bytes, 0), shift);
        table[prev_hash] = ip - base_ip - 1;
        uint32 cur_hash = HashBytes(GetUint32AtOffset(input_bytes, 1), shift);
        candidate = base_ip + table[cur_hash];
        candidate_bytes = UNALIGNED_LOAD32(candidate);
        table[cur_hash] = ip - base_ip;
      } while (GetUint32AtOffset(input_bytes, 1) == candidate_bytes);

      next_hash = HashBytes(GetUint32AtOffset(input_bytes, 2), shift);
      ++ip;
    }
  }

 emit_remainder:
  // Emit the remaining bytes as a literal
  if (next_emit < ip_end) {
    op = EmitLiteral(op, next_emit, ip_end - next_emit, false);
  }

  return op;
}
}  // end namespace internal

// Signature of output types needed by decompression code.
// The decompression code is templatized on a type that obeys this
// signature so that we do not pay virtual function call overhead in
// the middle of a tight decompression loop.
//
// class DecompressionWriter {
//  public:
//   // Called before decompression
//   void SetExpectedLength(size_t length);
//
//   // Called after decompression
//   bool CheckLength() const;
//
//   // Called repeatedly during decompression
//   bool Append(const char* ip, size_t length);
//   bool AppendFromSelf(uint32 offset, size_t length);
//
//   // The rules for how TryFastAppend differs from Append are somewhat
//   // convoluted:
//   //
//   //  - TryFastAppend is allowed to decline (return false) at any
//   //    time, for any reason -- just "return false" would be
//   //    a perfectly legal implementation of TryFastAppend.
//   //    The intention is for TryFastAppend to allow a fast path
//   //    in the common case of a small append.
//   //  - TryFastAppend is allowed to read up to <available> bytes
//   //    from the input buffer, whereas Append is allowed to read
//   //    <length>. However, if it returns true, it must leave
//   //    at least five (kMaximumTagLength) bytes in the input buffer
//   //    afterwards, so that there is always enough space to read the
//   //    next tag without checking for a refill.
//   //  - TryFastAppend must always return decline (return false)
//   //    if <length> is 61 or more, as in this case the literal length is not
//   //    decoded fully. In practice, this should not be a big problem,
//   //    as it is unlikely that one would implement a fast path accepting
//   //    this much data.
//   //
//   bool TryFastAppend(const char* ip, size_t available, size_t length);
// };

// -----------------------------------------------------------------------
// Lookup table for decompression code.  Generated by ComputeTable() below.
// -----------------------------------------------------------------------

// Mapping from i in range [0,4] to a mask to extract the bottom 8*i bits
static const uint32 wordmask[] = {
  0u, 0xffu, 0xffffu, 0xffffffu, 0xffffffffu
};

// Data stored per entry in lookup table:
//      Range   Bits-used       Description
//      ------------------------------------
//      1..64   0..7            Literal/copy length encoded in opcode byte
//      0..7    8..10           Copy offset encoded in opcode byte / 256
//      0..4    11..13          Extra bytes after opcode
//
// We use eight bits for the length even though 7 would have sufficed
// because of efficiency reasons:
//      (1) Extracting a byte is faster than a bit-field
//      (2) It properly aligns copy offset so we do not need a <<8
static const uint16 char_table[256] = {
  0x0001, 0x0804, 0x1001, 0x2001, 0x0002, 0x0805, 0x1002, 0x2002,
  0x0003, 0x0806, 0x1003, 0x2003, 0x0004, 0x0807, 0x1004, 0x2004,
  0x0005, 0x0808, 0x1005, 0x2005, 0x0006, 0x0809, 0x1006, 0x2006,
  0x0007, 0x080a, 0x1007, 0x2007, 0x0008, 0x080b, 0x1008, 0x2008,
  0x0009, 0x0904, 0x1009, 0x2009, 0x000a, 0x0905, 0x100a, 0x200a,
  0x000b, 0x0906, 0x100b, 0x200b, 0x000c, 0x0907, 0x100c, 0x200c,
  0x000d, 0x0908, 0x100d, 0x200d, 0x000e, 0x0909, 0x100e, 0x200e,
  0x000f, 0x090a, 0x100f, 0x200f, 0x0010, 0x090b, 0x1010, 0x2010,
  0x0011, 0x0a04, 0x1011, 0x2011, 0x0012, 0x0a05, 0x1012, 0x2012,
  0x0013, 0x0a06, 0x1013, 0x2013, 0x0014, 0x0a07, 0x1014, 0x2014,
  0x0015, 0x0a08, 0x1015, 0x2015, 0x0016, 0x0a09, 0x1016, 0x2016,
  0x0017, 0x0a0a, 0x1017, 0x2017, 0x0018, 0x0a0b, 0x1018, 0x2018,
  0x0019, 0x0b04, 0x1019, 0x2019, 0x001a, 0x0b05, 0x101a, 0x201a,
  0x001b, 0x0b06, 0x101b, 0x201b, 0x001c, 0x0b07, 0x101c, 0x201c,
  0x001d, 0x0b08, 0x101d, 0x201d, 0x001e, 0x0b09, 0x101e, 0x201e,
  0x001f, 0x0b0a, 0x101f, 0x201f, 0x0020, 0x0b0b, 0x1020, 0x2020,
  0x0021, 0x0c04, 0x1021, 0x2021, 0x0022, 0x0c05, 0x1022, 0x2022,
  0x0023, 0x0c06, 0x1023, 0x2023, 0x0024, 0x0c07, 0x1024, 0x2024,
  0x0025, 0x0c08, 0x1025, 0x2025, 0x0026, 0x0c09, 0x1026, 0x2026,
  0x0027, 0x0c0a, 0x1027, 0x2027, 0x0028, 0x0c0b, 0x1028, 0x2028,
  0x0029, 0x0d04, 0x1029, 0x2029, 0x002a, 0x0d05, 0x102a, 0x202a,
  0x002b, 0x0d06, 0x102b, 0x202b, 0x002c, 0x0d07, 0x102c, 0x202c,
  0x002d, 0x0d08, 0x102d, 0x202d, 0x002e, 0x0d09, 0x102e, 0x202e,
  0x002f, 0x0d0a, 0x102f, 0x202f, 0x0030, 0x0d0b, 0x1030, 0x2030,
  0x0031, 0x0e04, 0x1031, 0x2031, 0x0032, 0x0e05, 0x1032, 0x2032,
  0x0033, 0x0e06, 0x1033, 0x2033, 0x0034, 0x0e07, 0x1034, 0x2034,
  0x0035, 0x0e08, 0x1035, 0x2035, 0x0036, 0x0e09, 0x1036, 0x2036,
  0x0037, 0x0e0a, 0x1037, 0x2037, 0x0038, 0x0e0b, 0x1038, 0x2038,
  0x0039, 0x0f04, 0x1039, 0x2039, 0x003a, 0x0f05, 0x103a, 0x203a,
  0x003b, 0x0f06, 0x103b, 0x203b, 0x003c, 0x0f07, 0x103c, 0x203c,
  0x0801, 0x0f08, 0x103d, 0x203d, 0x1001, 0x0f09, 0x103e, 0x203e,
  0x1801, 0x0f0a, 0x103f, 0x203f, 0x2001, 0x0f0b, 0x1040, 0x2040
};

// In debug mode, allow optional computation of the table at startup.
// Also, check that the decompression table is correct.
#ifndef NDEBUG
DEFINE_bool(snappy_dump_decompression_table, false,
            "If true, we print the decompression table at startup.");

static uint16 MakeEntry(unsigned int extra,
                        unsigned int len,
                        unsigned int copy_offset) {
  // Check that all of the fields fit within the allocated space
  assert(extra       == (extra & 0x7));          // At most 3 bits
  assert(copy_offset == (copy_offset & 0x7));    // At most 3 bits
  assert(len         == (len & 0x7f));           // At most 7 bits
  return len | (copy_offset << 8) | (extra << 11);
}

static void ComputeTable() {
  uint16 dst[256];

  // Place invalid entries in all places to detect missing initialization
  int assigned = 0;
  for (int i = 0; i < 256; i++) {
    dst[i] = 0xffff;
  }

  // Small LITERAL entries.  We store (len-1) in the top 6 bits.
  for (unsigned int len = 1; len <= 60; len++) {
    dst[LITERAL | ((len-1) << 2)] = MakeEntry(0, len, 0);
    assigned++;
  }

  // Large LITERAL entries.  We use 60..63 in the high 6 bits to
  // encode the number of bytes of length info that follow the opcode.
  for (unsigned int extra_bytes = 1; extra_bytes <= 4; extra_bytes++) {
    // We set the length field in the lookup table to 1 because extra
    // bytes encode len-1.
    dst[LITERAL | ((extra_bytes+59) << 2)] = MakeEntry(extra_bytes, 1, 0);
    assigned++;
  }

  // COPY_1_BYTE_OFFSET.
  //
  // The tag byte in the compressed data stores len-4 in 3 bits, and
  // offset/256 in 5 bits.  offset%256 is stored in the next byte.
  //
  // This format is used for length in range [4..11] and offset in
  // range [0..2047]
  for (unsigned int len = 4; len < 12; len++) {
    for (unsigned int offset = 0; offset < 2048; offset += 256) {
      dst[COPY_1_BYTE_OFFSET | ((len-4)<<2) | ((offset>>8)<<5)] =
        MakeEntry(1, len, offset>>8);
      assigned++;
    }
  }

  // COPY_2_BYTE_OFFSET.
  // Tag contains len-1 in top 6 bits, and offset in next two bytes.
  for (unsigned int len = 1; len <= 64; len++) {
    dst[COPY_2_BYTE_OFFSET | ((len-1)<<2)] = MakeEntry(2, len, 0);
    assigned++;
  }

  // COPY_4_BYTE_OFFSET.
  // Tag contents len-1 in top 6 bits, and offset in next four bytes.
  for (unsigned int len = 1; len <= 64; len++) {
    dst[COPY_4_BYTE_OFFSET | ((len-1)<<2)] = MakeEntry(4, len, 0);
    assigned++;
  }

  // Check that each entry was initialized exactly once.
  if (assigned != 256) {
    fprintf(stderr, "ComputeTable: assigned only %d of 256\n", assigned);
    abort();
  }
  for (int i = 0; i < 256; i++) {
    if (dst[i] == 0xffff) {
      fprintf(stderr, "ComputeTable: did not assign byte %d\n", i);
      abort();
    }
  }

  if (FLAGS_snappy_dump_decompression_table) {
    printf("static const uint16 char_table[256] = {\n  ");
    for (int i = 0; i < 256; i++) {
      printf("0x%04x%s",
             dst[i],
             ((i == 255) ? "\n" : (((i%8) == 7) ? ",\n  " : ", ")));
    }
    printf("};\n");
  }

  // Check that computed table matched recorded table
  for (int i = 0; i < 256; i++) {
    if (dst[i] != char_table[i]) {
      fprintf(stderr, "ComputeTable: byte %d: computed (%x), expect (%x)\n",
              i, static_cast<int>(dst[i]), static_cast<int>(char_table[i]));
      abort();
    }
  }
}
#endif /* !NDEBUG */

// Helper class for decompression
class SnappyDecompressor {
 private:
  Source*       reader_;         // Underlying source of bytes to decompress
  const char*   ip_;             // Points to next buffered byte
  const char*   ip_limit_;       // Points just past buffered bytes
  uint32        peeked_;         // Bytes peeked from reader (need to skip)
  bool          eof_;            // Hit end of input without an error?
  char          scratch_[kMaximumTagLength];  // See RefillTag().

  // Ensure that all of the tag metadata for the next tag is available
  // in [ip_..ip_limit_-1].  Also ensures that [ip,ip+4] is readable even
  // if (ip_limit_ - ip_ < 5).
  //
  // Returns true on success, false on error or end of input.
  bool RefillTag();

 public:
  explicit SnappyDecompressor(Source* reader)
      : reader_(reader),
        ip_(NULL),
        ip_limit_(NULL),
        peeked_(0),
        eof_(false) {
  }

  ~SnappyDecompressor() {
    // Advance past any bytes we peeked at from the reader
    reader_->Skip(peeked_);
  }

  // Returns true iff we have hit the end of the input without an error.
  bool eof() const {
    return eof_;
  }

  // Read the uncompressed length stored at the start of the compressed data.
  // On succcess, stores the length in *result and returns true.
  // On failure, returns false.
  bool ReadUncompressedLength(uint32* result) {
    assert(ip_ == NULL);       // Must not have read anything yet
    // Length is encoded in 1..5 bytes
    *result = 0;
    uint32 shift = 0;
    while (true) {
      if (shift >= 32) return false;
      size_t n;
      const char* ip = reader_->Peek(&n);
      if (n == 0) return false;
      const unsigned char c = *(reinterpret_cast<const unsigned char*>(ip));
      reader_->Skip(1);
      *result |= static_cast<uint32>(c & 0x7f) << shift;
      if (c < 128) {
        break;
      }
      shift += 7;
    }
    return true;
  }

  // Process the next item found in the input.
  // Returns true if successful, false on error or end of input.
  template <class Writer>
  void DecompressAllTags(Writer* writer) {
    const char* ip = ip_;

    // We could have put this refill fragment only at the beginning of the loop.
    // However, duplicating it at the end of each branch gives the compiler more
    // scope to optimize the <ip_limit_ - ip> expression based on the local
    // context, which overall increases speed.
    #define MAYBE_REFILL() \
        if (ip_limit_ - ip < kMaximumTagLength) { \
          ip_ = ip; \
          if (!RefillTag()) return; \
          ip = ip_; \
        }

    MAYBE_REFILL();
    for ( ;; ) {
      const unsigned char c = *(reinterpret_cast<const unsigned char*>(ip++));

      if ((c & 0x3) == LITERAL) {
        size_t literal_length = (c >> 2) + 1u;
        if (writer->TryFastAppend(ip, ip_limit_ - ip, literal_length)) {
          assert(literal_length < 61);
          ip += literal_length;
          // NOTE(user): There is no MAYBE_REFILL() here, as TryFastAppend()
          // will not return true unless there's already at least five spare
          // bytes in addition to the literal.
          continue;
        }
        if (PREDICT_FALSE(literal_length >= 61)) {
          // Long literal.
          const size_t literal_length_length = literal_length - 60;
          literal_length =
              (LittleEndian::Load32(ip) & wordmask[literal_length_length]) + 1;
          ip += literal_length_length;
        }

        size_t avail = ip_limit_ - ip;
        while (avail < literal_length) {
          if (!writer->Append(ip, avail)) return;
          literal_length -= avail;
          reader_->Skip(peeked_);
          size_t n;
          ip = reader_->Peek(&n);
          avail = n;
          peeked_ = avail;
          if (avail == 0) return;  // Premature end of input
          ip_limit_ = ip + avail;
        }
        if (!writer->Append(ip, literal_length)) {
          return;
        }
        ip += literal_length;
        MAYBE_REFILL();
      } else {
        const uint32 entry = char_table[c];
        const uint32 trailer = LittleEndian::Load32(ip) & wordmask[entry >> 11];
        const uint32 length = entry & 0xff;
        ip += entry >> 11;

        // copy_offset/256 is encoded in bits 8..10.  By just fetching
        // those bits, we get copy_offset (since the bit-field starts at
        // bit 8).
        const uint32 copy_offset = entry & 0x700;
        if (!writer->AppendFromSelf(copy_offset + trailer, length)) {
          return;
        }
        MAYBE_REFILL();
      }
    }

#undef MAYBE_REFILL
  }
};

bool SnappyDecompressor::RefillTag() {
  const char* ip = ip_;
  if (ip == ip_limit_) {
    // Fetch a new fragment from the reader
    reader_->Skip(peeked_);   // All peeked bytes are used up
    size_t n;
    ip = reader_->Peek(&n);
    peeked_ = n;
    if (n == 0) {
      eof_ = true;
      return false;
    }
    ip_limit_ = ip + n;
  }

  // Read the tag character
  assert(ip < ip_limit_);
  const unsigned char c = *(reinterpret_cast<const unsigned char*>(ip));
  const uint32 entry = char_table[c];
  const uint32 needed = (entry >> 11) + 1;  // +1 byte for 'c'
  assert(needed <= sizeof(scratch_));

  // Read more bytes from reader if needed
  uint32 nbuf = ip_limit_ - ip;
  if (nbuf < needed) {
    // Stitch together bytes from ip and reader to form the word
    // contents.  We store the needed bytes in "scratch_".  They
    // will be consumed immediately by the caller since we do not
    // read more than we need.
    memmove(scratch_, ip, nbuf);
    reader_->Skip(peeked_);  // All peeked bytes are used up
    peeked_ = 0;
    while (nbuf < needed) {
      size_t length;
      const char* src = reader_->Peek(&length);
      if (length == 0) return false;
      uint32 to_add = min<uint32>(needed - nbuf, length);
      memcpy(scratch_ + nbuf, src, to_add);
      nbuf += to_add;
      reader_->Skip(to_add);
    }
    assert(nbuf == needed);
    ip_ = scratch_;
    ip_limit_ = scratch_ + needed;
  } else if (nbuf < kMaximumTagLength) {
    // Have enough bytes, but move into scratch_ so that we do not
    // read past end of input
    memmove(scratch_, ip, nbuf);
    reader_->Skip(peeked_);  // All peeked bytes are used up
    peeked_ = 0;
    ip_ = scratch_;
    ip_limit_ = scratch_ + nbuf;
  } else {
    // Pass pointer to buffer returned by reader_.
    ip_ = ip;
  }
  return true;
}

template <typename Writer>
static bool InternalUncompress(Source* r, Writer* writer) {
  // Read the uncompressed length from the front of the compressed input
  SnappyDecompressor decompressor(r);
  uint32 uncompressed_len = 0;
  if (!decompressor.ReadUncompressedLength(&uncompressed_len)) return false;
  return InternalUncompressAllTags(&decompressor, writer, uncompressed_len);
}

template <typename Writer>
static bool InternalUncompressAllTags(SnappyDecompressor* decompressor,
                                      Writer* writer,
                                      uint32 uncompressed_len) {
  writer->SetExpectedLength(uncompressed_len);

  // Process the entire input
  decompressor->DecompressAllTags(writer);
  return (decompressor->eof() && writer->CheckLength());
}

bool GetUncompressedLength(Source* source, uint32* result) {
  SnappyDecompressor decompressor(source);
  return decompressor.ReadUncompressedLength(result);
}

size_t Compress(Source* reader, Sink* writer) {
  size_t written = 0;
  size_t N = reader->Available();
  char ulength[Varint::kMax32];
  char* p = Varint::Encode32(ulength, N);
  writer->Append(ulength, p-ulength);
  written += (p - ulength);

  internal::WorkingMemory wmem;
  char* scratch = NULL;
  char* scratch_output = NULL;

  while (N > 0) {
    // Get next block to compress (without copying if possible)
    size_t fragment_size;
    const char* fragment = reader->Peek(&fragment_size);
    assert(fragment_size != 0);  // premature end of input
    const size_t num_to_read = min(N, kBlockSize);
    size_t bytes_read = fragment_size;

    size_t pending_advance = 0;
    if (bytes_read >= num_to_read) {
      // Buffer returned by reader is large enough
      pending_advance = num_to_read;
      fragment_size = num_to_read;
    } else {
      // Read into scratch buffer
      if (scratch == NULL) {
        // If this is the last iteration, we want to allocate N bytes
        // of space, otherwise the max possible kBlockSize space.
        // num_to_read contains exactly the correct value
        scratch = new char[num_to_read];
      }
      memcpy(scratch, fragment, bytes_read);
      reader->Skip(bytes_read);

      while (bytes_read < num_to_read) {
        fragment = reader->Peek(&fragment_size);
        size_t n = min<size_t>(fragment_size, num_to_read - bytes_read);
        memcpy(scratch + bytes_read, fragment, n);
        bytes_read += n;
        reader->Skip(n);
      }
      assert(bytes_read == num_to_read);
      fragment = scratch;
      fragment_size = num_to_read;
    }
    assert(fragment_size == num_to_read);

    // Get encoding table for compression
    int table_size;
    uint16* table = wmem.GetHashTable(num_to_read, &table_size);

    // Compress input_fragment and append to dest
    const int max_output = MaxCompressedLength(num_to_read);

    // Need a scratch buffer for the output, in case the byte sink doesn't
    // have room for us directly.
    if (scratch_output == NULL) {
      scratch_output = new char[max_output];
    } else {
      // Since we encode kBlockSize regions followed by a region
      // which is <= kBlockSize in length, a previously allocated
      // scratch_output[] region is big enough for this iteration.
    }
    char* dest = writer->GetAppendBuffer(max_output, scratch_output);
    char* end = internal::CompressFragment(fragment, fragment_size,
                                           dest, table, table_size);
    writer->Append(dest, end - dest);
    written += (end - dest);

    N -= num_to_read;
    reader->Skip(pending_advance);
  }

  delete[] scratch;
  delete[] scratch_output;

  return written;
}

// -----------------------------------------------------------------------
// IOVec interfaces
// -----------------------------------------------------------------------

// A type that writes to an iovec.
// Note that this is not a "ByteSink", but a type that matches the
// Writer template argument to SnappyDecompressor::DecompressAllTags().
class SnappyIOVecWriter {
 private:
  const struct iovec* output_iov_;
  const size_t output_iov_count_;

  // We are currently writing into output_iov_[curr_iov_index_].
  int curr_iov_index_;

  // Bytes written to output_iov_[curr_iov_index_] so far.
  size_t curr_iov_written_;

  // Total bytes decompressed into output_iov_ so far.
  size_t total_written_;

  // Maximum number of bytes that will be decompressed into output_iov_.
  size_t output_limit_;

  inline char* GetIOVecPointer(int index, size_t offset) {
    return reinterpret_cast<char*>(output_iov_[index].iov_base) +
        offset;
  }

 public:
  // Does not take ownership of iov. iov must be valid during the
  // entire lifetime of the SnappyIOVecWriter.
  inline SnappyIOVecWriter(const struct iovec* iov, size_t iov_count)
      : output_iov_(iov),
        output_iov_count_(iov_count),
        curr_iov_index_(0),
        curr_iov_written_(0),
        total_written_(0),
        output_limit_(-1) {
  }

  inline void SetExpectedLength(size_t len) {
    output_limit_ = len;
  }

  inline bool CheckLength() const {
    return total_written_ == output_limit_;
  }

  inline bool Append(const char* ip, size_t len) {
    if (total_written_ + len > output_limit_) {
      return false;
    }

    while (len > 0) {
      assert(curr_iov_written_ <= output_iov_[curr_iov_index_].iov_len);
      if (curr_iov_written_ >= output_iov_[curr_iov_index_].iov_len) {
        // This iovec is full. Go to the next one.
        if (curr_iov_index_ + 1 >= output_iov_count_) {
          return false;
        }
        curr_iov_written_ = 0;
        ++curr_iov_index_;
      }

	  const size_t to_write = std::min(
          len, output_iov_[curr_iov_index_].iov_len - curr_iov_written_);
      memcpy(GetIOVecPointer(curr_iov_index_, curr_iov_written_),
             ip,
             to_write);
      curr_iov_written_ += to_write;
      total_written_ += to_write;
      ip += to_write;
      len -= to_write;
    }

    return true;
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t len) {
    const size_t space_left = output_limit_ - total_written_;
    if (len <= 16 && available >= 16 + kMaximumTagLength && space_left >= 16 &&
        output_iov_[curr_iov_index_].iov_len - curr_iov_written_ >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      char* ptr = GetIOVecPointer(curr_iov_index_, curr_iov_written_);
      UnalignedCopy64(ip, ptr);
      UnalignedCopy64(ip + 8, ptr + 8);
      curr_iov_written_ += len;
      total_written_ += len;
      return true;
    }

    return false;
  }

  inline bool AppendFromSelf(size_t offset, size_t len) {
    if (offset > total_written_ || offset == 0) {
      return false;
    }
    const size_t space_left = output_limit_ - total_written_;
    if (len > space_left) {
      return false;
    }

    // Locate the iovec from which we need to start the copy.
    int from_iov_index = curr_iov_index_;
    size_t from_iov_offset = curr_iov_written_;
    while (offset > 0) {
      if (from_iov_offset >= offset) {
        from_iov_offset -= offset;
        break;
      }

      offset -= from_iov_offset;
      --from_iov_index;
      assert(from_iov_index >= 0);
      from_iov_offset = output_iov_[from_iov_index].iov_len;
    }

    // Copy <len> bytes starting from the iovec pointed to by from_iov_index to
    // the current iovec.
    while (len > 0) {
      assert(from_iov_index <= curr_iov_index_);
      if (from_iov_index != curr_iov_index_) {
        const size_t to_copy = std::min(
            output_iov_[from_iov_index].iov_len - from_iov_offset,
            len);
        Append(GetIOVecPointer(from_iov_index, from_iov_offset), to_copy);
        len -= to_copy;
        if (len > 0) {
          ++from_iov_index;
          from_iov_offset = 0;
        }
      } else {
        assert(curr_iov_written_ <= output_iov_[curr_iov_index_].iov_len);
        size_t to_copy = std::min(output_iov_[curr_iov_index_].iov_len -
                                      curr_iov_written_,
                                  len);
        if (to_copy == 0) {
          // This iovec is full. Go to the next one.
          if (curr_iov_index_ + 1 >= output_iov_count_) {
            return false;
          }
          ++curr_iov_index_;
          curr_iov_written_ = 0;
          continue;
        }
        if (to_copy > len) {
          to_copy = len;
        }
        IncrementalCopy(GetIOVecPointer(from_iov_index, from_iov_offset),
                        GetIOVecPointer(curr_iov_index_, curr_iov_written_),
                        to_copy);
        curr_iov_written_ += to_copy;
        from_iov_offset += to_copy;
        total_written_ += to_copy;
        len -= to_copy;
      }
    }

    return true;
  }

};

bool RawUncompressToIOVec(const char* compressed, size_t compressed_length,
                          const struct iovec* iov, size_t iov_cnt) {
  ByteArraySource reader(compressed, compressed_length);
  return RawUncompressToIOVec(&reader, iov, iov_cnt);
}

bool RawUncompressToIOVec(Source* compressed, const struct iovec* iov,
                          size_t iov_cnt) {
  SnappyIOVecWriter output(iov, iov_cnt);
  return InternalUncompress(compressed, &output);
}

// -----------------------------------------------------------------------
// Flat array interfaces
// -----------------------------------------------------------------------

// A type that writes to a flat array.
// Note that this is not a "ByteSink", but a type that matches the
// Writer template argument to SnappyDecompressor::DecompressAllTags().
class SnappyArrayWriter {
 private:
  char* base_;
  char* op_;
  char* op_limit_;

 public:
  inline explicit SnappyArrayWriter(char* dst)
      : base_(dst),
        op_(dst) {
  }

  inline void SetExpectedLength(size_t len) {
    op_limit_ = op_ + len;
  }

  inline bool CheckLength() const {
    return op_ == op_limit_;
  }

  inline bool Append(const char* ip, size_t len) {
    char* op = op_;
    const size_t space_left = op_limit_ - op;
    if (space_left < len) {
      return false;
    }
    memcpy(op, ip, len);
    op_ = op + len;
    return true;
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t len) {
    char* op = op_;
    const size_t space_left = op_limit_ - op;
    if (len <= 16 && available >= 16 + kMaximumTagLength && space_left >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      UnalignedCopy64(ip, op);
      UnalignedCopy64(ip + 8, op + 8);
      op_ = op + len;
      return true;
    } else {
      return false;
    }
  }

  inline bool AppendFromSelf(size_t offset, size_t len) {
    char* op = op_;
    const size_t space_left = op_limit_ - op;

    // Check if we try to append from before the start of the buffer.
    // Normally this would just be a check for "produced < offset",
    // but "produced <= offset - 1u" is equivalent for every case
    // except the one where offset==0, where the right side will wrap around
    // to a very big number. This is convenient, as offset==0 is another
    // invalid case that we also want to catch, so that we do not go
    // into an infinite loop.
    assert(op >= base_);
    size_t produced = op - base_;
    if (produced <= offset - 1u) {
      return false;
    }
    if (len <= 16 && offset >= 8 && space_left >= 16) {
      // Fast path, used for the majority (70-80%) of dynamic invocations.
      UnalignedCopy64(op - offset, op);
      UnalignedCopy64(op - offset + 8, op + 8);
    } else {
      if (space_left >= len + kMaxIncrementCopyOverflow) {
        IncrementalCopyFastPath(op - offset, op, len);
      } else {
        if (space_left < len) {
          return false;
        }
        IncrementalCopy(op - offset, op, len);
      }
    }

    op_ = op + len;
    return true;
  }
};

bool RawUncompress(const char* compressed, size_t n, char* uncompressed) {
  ByteArraySource reader(compressed, n);
  return RawUncompress(&reader, uncompressed);
}

bool RawUncompress(Source* compressed, char* uncompressed) {
  SnappyArrayWriter output(uncompressed);
  return InternalUncompress(compressed, &output);
}

bool Uncompress(const char* compressed, size_t n, string* uncompressed) {
  size_t ulength;
  if (!GetUncompressedLength(compressed, n, &ulength)) {
    return false;
  }
  // On 32-bit builds: max_size() < kuint32max.  Check for that instead
  // of crashing (e.g., consider externally specified compressed data).
  if (ulength > uncompressed->max_size()) {
    return false;
  }
  STLStringResizeUninitialized(uncompressed, ulength);
  return RawUncompress(compressed, n, string_as_array(uncompressed));
}


// A Writer that drops everything on the floor and just does validation
class SnappyDecompressionValidator {
 private:
  size_t expected_;
  size_t produced_;

 public:
  inline SnappyDecompressionValidator() : produced_(0) { }
  inline void SetExpectedLength(size_t len) {
    expected_ = len;
  }
  inline bool CheckLength() const {
    return expected_ == produced_;
  }
  inline bool Append(const char* ip, size_t len) {
    produced_ += len;
    return produced_ <= expected_;
  }
  inline bool TryFastAppend(const char* ip, size_t available, size_t length) {
    return false;
  }
  inline bool AppendFromSelf(size_t offset, size_t len) {
    // See SnappyArrayWriter::AppendFromSelf for an explanation of
    // the "offset - 1u" trick.
    if (produced_ <= offset - 1u) return false;
    produced_ += len;
    return produced_ <= expected_;
  }
};

bool IsValidCompressedBuffer(const char* compressed, size_t n) {
  ByteArraySource reader(compressed, n);
  SnappyDecompressionValidator writer;
  return InternalUncompress(&reader, &writer);
}

void RawCompress(const char* input,
                 size_t input_length,
                 char* compressed,
                 size_t* compressed_length) {
  ByteArraySource reader(input, input_length);
  UncheckedByteArraySink writer(compressed);
  Compress(&reader, &writer);

  // Compute how many bytes were added
  *compressed_length = (writer.CurrentDestination() - compressed);
}

size_t Compress(const char* input, size_t input_length, string* compressed) {
  // Pre-grow the buffer to the max length of the compressed output
  compressed->resize(MaxCompressedLength(input_length));

  size_t compressed_length;
  RawCompress(input, input_length, string_as_array(compressed),
              &compressed_length);
  compressed->resize(compressed_length);
  return compressed_length;
}


} // end namespace snappy


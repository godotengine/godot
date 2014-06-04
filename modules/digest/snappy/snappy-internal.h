// Copyright 2008 Google Inc. All Rights Reserved.
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
//
// Internals shared between the Snappy implementation and its unittest.

#ifndef UTIL_SNAPPY_SNAPPY_INTERNAL_H_
#define UTIL_SNAPPY_SNAPPY_INTERNAL_H_

#include "snappy-stubs-internal.h"

namespace snappy {
namespace internal {

class WorkingMemory {
 public:
  WorkingMemory() : large_table_(NULL) { }
  ~WorkingMemory() { delete[] large_table_; }

  // Allocates and clears a hash table using memory in "*this",
  // stores the number of buckets in "*table_size" and returns a pointer to
  // the base of the hash table.
  uint16* GetHashTable(size_t input_size, int* table_size);

 private:
  uint16 small_table_[1<<10];    // 2KB
  uint16* large_table_;          // Allocated only when needed

  DISALLOW_COPY_AND_ASSIGN(WorkingMemory);
};

// Flat array compression that does not emit the "uncompressed length"
// prefix. Compresses "input" string to the "*op" buffer.
//
// REQUIRES: "input_length <= kBlockSize"
// REQUIRES: "op" points to an array of memory that is at least
// "MaxCompressedLength(input_length)" in size.
// REQUIRES: All elements in "table[0..table_size-1]" are initialized to zero.
// REQUIRES: "table_size" is a power of two
//
// Returns an "end" pointer into "op" buffer.
// "end - op" is the compressed size of "input".
char* CompressFragment(const char* input,
                       size_t input_length,
                       char* op,
                       uint16* table,
                       const int table_size);

// Return the largest n such that
//
//   s1[0,n-1] == s2[0,n-1]
//   and n <= (s2_limit - s2).
//
// Does not read *s2_limit or beyond.
// Does not read *(s1 + (s2_limit - s2)) or beyond.
// Requires that s2_limit >= s2.
//
// Separate implementation for x86_64, for speed.  Uses the fact that
// x86_64 is little endian.
#if defined(ARCH_K8)
static inline int FindMatchLength(const char* s1,
                                  const char* s2,
                                  const char* s2_limit) {
  assert(s2_limit >= s2);
  int matched = 0;

  // Find out how long the match is. We loop over the data 64 bits at a
  // time until we find a 64-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (PREDICT_TRUE(s2 <= s2_limit - 8)) {
    if (PREDICT_FALSE(UNALIGNED_LOAD64(s2) == UNALIGNED_LOAD64(s1 + matched))) {
      s2 += 8;
      matched += 8;
    } else {
      // On current (mid-2008) Opteron models there is a 3% more
      // efficient code sequence to find the first non-matching byte.
      // However, what follows is ~10% better on Intel Core 2 and newer,
      // and we expect AMD's bsf instruction to improve.
      uint64 x = UNALIGNED_LOAD64(s2) ^ UNALIGNED_LOAD64(s1 + matched);
      int matching_bits = Bits::FindLSBSetNonZero64(x);
      matched += matching_bits >> 3;
      return matched;
    }
  }
  while (PREDICT_TRUE(s2 < s2_limit)) {
    if (PREDICT_TRUE(s1[matched] == *s2)) {
      ++s2;
      ++matched;
    } else {
      return matched;
    }
  }
  return matched;
}
#else
static inline int FindMatchLength(const char* s1,
                                  const char* s2,
                                  const char* s2_limit) {
  // Implementation based on the x86-64 version, above.
  assert(s2_limit >= s2);
  int matched = 0;

  while (s2 <= s2_limit - 4 &&
         UNALIGNED_LOAD32(s2) == UNALIGNED_LOAD32(s1 + matched)) {
    s2 += 4;
    matched += 4;
  }
  if (LittleEndian::IsLittleEndian() && s2 <= s2_limit - 4) {
    uint32 x = UNALIGNED_LOAD32(s2) ^ UNALIGNED_LOAD32(s1 + matched);
    int matching_bits = Bits::FindLSBSetNonZero(x);
    matched += matching_bits >> 3;
  } else {
    while ((s2 < s2_limit) && (s1[matched] == *s2)) {
      ++s2;
      ++matched;
    }
  }
  return matched;
}
#endif

}  // end namespace internal
}  // end namespace snappy

#endif  // UTIL_SNAPPY_SNAPPY_INTERNAL_H_

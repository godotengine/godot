/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "gtest/gtest.h"

#include "test/acm_random.h"
#include "vp8/decoder/dboolhuff.h"
#include "vp8/encoder/boolhuff.h"
#include "vpx/vpx_integer.h"

namespace {
const int num_tests = 10;

// In a real use the 'decrypt_state' parameter will be a pointer to a struct
// with whatever internal state the decryptor uses. For testing we'll just
// xor with a constant key, and decrypt_state will point to the start of
// the original buffer.
const uint8_t secret_key[16] = {
  0x01, 0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78,
  0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0
};

void encrypt_buffer(uint8_t *buffer, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    buffer[i] ^= secret_key[i & 15];
  }
}

void test_decrypt_cb(void *decrypt_state, const uint8_t *input, uint8_t *output,
                     int count) {
  const size_t offset = input - reinterpret_cast<uint8_t *>(decrypt_state);
  for (int i = 0; i < count; i++) {
    output[i] = input[i] ^ secret_key[(offset + i) & 15];
  }
}

}  // namespace

using libvpx_test::ACMRandom;

TEST(VP8, TestBitIO) {
  ACMRandom rnd(ACMRandom::DeterministicSeed());
  for (int n = 0; n < num_tests; ++n) {
    for (int method = 0; method <= 7; ++method) {  // we generate various proba
      const int kBitsToTest = 1000;
      uint8_t probas[kBitsToTest];

      for (int i = 0; i < kBitsToTest; ++i) {
        const int parity = i & 1;
        /* clang-format off */
        probas[i] =
            (method == 0) ? 0 : (method == 1) ? 255 :
            (method == 2) ? 128 :
            (method == 3) ? rnd.Rand8() :
            (method == 4) ? (parity ? 0 : 255) :
            // alternate between low and high proba:
            (method == 5) ? (parity ? rnd(128) : 255 - rnd(128)) :
            (method == 6) ?
                (parity ? rnd(64) : 255 - rnd(64)) :
                (parity ? rnd(32) : 255 - rnd(32));
        /* clang-format on */
      }
      for (int bit_method = 0; bit_method <= 3; ++bit_method) {
        const int random_seed = 6432;
        const int kBufferSize = 10000;
        ACMRandom bit_rnd(random_seed);
        BOOL_CODER bw;
        uint8_t bw_buffer[kBufferSize];
        vp8_start_encode(&bw, bw_buffer, bw_buffer + kBufferSize);

        int bit = (bit_method == 0) ? 0 : (bit_method == 1) ? 1 : 0;
        for (int i = 0; i < kBitsToTest; ++i) {
          if (bit_method == 2) {
            bit = (i & 1);
          } else if (bit_method == 3) {
            bit = bit_rnd(2);
          }
          vp8_encode_bool(&bw, bit, static_cast<int>(probas[i]));
        }

        vp8_stop_encode(&bw);
        // vp8dx_bool_decoder_fill() may read into uninitialized data that
        // isn't used meaningfully, but may trigger an MSan warning.
        memset(bw_buffer + bw.pos, 0, sizeof(VP8_BD_VALUE) - 1);

        BOOL_DECODER br;
        encrypt_buffer(bw_buffer, kBufferSize);
        vp8dx_start_decode(&br, bw_buffer, kBufferSize, test_decrypt_cb,
                           reinterpret_cast<void *>(bw_buffer));
        bit_rnd.Reset(random_seed);
        for (int i = 0; i < kBitsToTest; ++i) {
          if (bit_method == 2) {
            bit = (i & 1);
          } else if (bit_method == 3) {
            bit = bit_rnd(2);
          }
          GTEST_ASSERT_EQ(vp8dx_decode_bool(&br, probas[i]), bit)
              << "pos: " << i << " / " << kBitsToTest
              << " bit_method: " << bit_method << " method: " << method;
        }
      }
    }
  }
}

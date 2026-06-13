// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ANS_PARAMS_H_
#define LIB_JXL_ANS_PARAMS_H_

// Common parameters that are needed for both the ANS entropy encoding and
// decoding methods.

#include <stdint.h>
#include <stdlib.h>

namespace jxl {

// TODO(veluca): decide if 12 is the best constant here (valid range is up to
// 16). This requires recomputing the Huffman tables in {enc,dec}_ans.cc
// 14 gives a 0.2% improvement at d1 and makes d8 slightly worse. This is
// likely not worth the increase in encoder complexity.
#define ANS_LOG_TAB_SIZE 12u
#define ANS_TAB_SIZE (1 << ANS_LOG_TAB_SIZE)
#define ANS_TAB_MASK (ANS_TAB_SIZE - 1)

// Largest possible symbol to be encoded by either ANS or prefix coding.
#define PREFIX_MAX_ALPHABET_SIZE 4096
#define ANS_MAX_ALPHABET_SIZE 256

// Max number of bits for prefix coding.
#define PREFIX_MAX_BITS 15

#define ANS_SIGNATURE 0x13  // Initial state, used as CRC.

}  // namespace jxl

#endif  // LIB_JXL_ANS_PARAMS_H_

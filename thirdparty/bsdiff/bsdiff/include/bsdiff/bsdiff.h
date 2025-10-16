// Copyright 2015 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_BSDIFF_H_
#define _BSDIFF_BSDIFF_H_

#include <stddef.h>
#include <stdint.h>

#include "bsdiff/common.h"
#include "bsdiff/patch_writer_interface.h"
#include "bsdiff/suffix_array_index_interface.h"

namespace bsdiff {

BSDIFF_EXPORT
int bsdiff(const uint8_t* old_buf,
           size_t oldsize,
           const uint8_t* new_buf,
           size_t newsize,
           PatchWriterInterface* patch,
           SuffixArrayIndexInterface** sai_cache);

// The |min_length| parameter determines the required minimum length of a match
// to be considered instead of emitting mismatches. The minimum value is 9,
// since smaller matches are always ignored. If a smaller value is passed, the
// minimum value of 9 will be used instead. A very large value (past 30) will
// give increasingly bad results as you increase the minimum length since legit
// matches between the old and new data will be ignored. The exact best value
// depends on the data, but the sweet spot should be between 9 and 20 for the
// examples tested.
BSDIFF_EXPORT
int bsdiff(const uint8_t* old_buf,
           size_t oldsize,
           const uint8_t* new_buf,
           size_t newsize,
           size_t min_length,
           PatchWriterInterface* patch,
           SuffixArrayIndexInterface** sai_cache);

}  // namespace bsdiff

#endif  // _BSDIFF_BSDIFF_H_

// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_SUFFIX_ARRAY_INDEX_H_
#define _BSDIFF_SUFFIX_ARRAY_INDEX_H_

#include <stdint.h>

#include <memory>

#include "bsdiff/suffix_array_index_interface.h"

namespace bsdiff {

std::unique_ptr<SuffixArrayIndexInterface> CreateSuffixArrayIndex(
    const uint8_t* text,
    size_t n);

}  // namespace bsdiff

#endif  // _BSDIFF_SUFFIX_ARRAY_INDEX_H_

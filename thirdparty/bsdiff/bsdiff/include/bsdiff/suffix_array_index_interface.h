// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef _BSDIFF_SUFFIX_ARRAY_INDEX_INTERFACE_H_
#define _BSDIFF_SUFFIX_ARRAY_INDEX_INTERFACE_H_

// The SuffixArrayIndexInterface encapsulates a search index based on a
// suffix-array with a common string search interface. The implementations of
// this index can vary on technical details, such as the size of the internal
// suffix array elements, which are not visible in this interface.

#include <stddef.h>
#include <stdint.h>

namespace bsdiff {

class SuffixArrayIndexInterface {
 public:
  virtual ~SuffixArrayIndexInterface() = default;

  // Search in the index the longest prefix of the string |target| of length
  // |length|. The length of the longest prefix found, which could be 0, is
  // stored in |out_length| and a position in the source text where this prefix
  // was found is store in |out_pos|.
  virtual void SearchPrefix(const uint8_t* target,
                            size_t length,
                            size_t* out_length,
                            uint64_t* out_pos) const = 0;

 protected:
  SuffixArrayIndexInterface() = default;
};

}  // namespace bsdiff

#endif  // _BSDIFF_SUFFIX_ARRAY_INDEX_INTERFACE_H_

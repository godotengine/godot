// Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "bsdiff/suffix_array_index.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <divsufsort.h>
#include <divsufsort64.h>

#include "bsdiff/logging.h"

namespace {

// libdivsufsort C++ overloaded functions used to allow calling the right
// implementation based on the pointer size.
int CallDivSufSort(const uint8_t* text, saidx_t* sa, size_t n) {
  return divsufsort(text, sa, n);
}
int CallDivSufSort(const uint8_t* text, saidx64_t* sa, size_t n) {
  return divsufsort64(text, sa, n);
}

saidx_t CallSaSearch(const uint8_t* text,
                     size_t text_size,
                     const uint8_t* pattern,
                     size_t pattern_size,
                     const saidx_t* sa,
                     size_t sa_size,
                     saidx_t* left) {
  return sa_search(text, text_size, pattern, pattern_size, sa, sa_size, left);
}
saidx64_t CallSaSearch(const uint8_t* text,
                       size_t text_size,
                       const uint8_t* pattern,
                       size_t pattern_size,
                       const saidx64_t* sa,
                       size_t sa_size,
                       saidx64_t* left) {
  return sa_search64(text, text_size, pattern, pattern_size, sa, sa_size, left);
}

}  // namespace

namespace bsdiff {

// The SAIDX template type must be either saidx_t or saidx64_t, which will
// depend on the maximum size of the input text needed.
template <typename SAIDX>
class SuffixArrayIndex : public SuffixArrayIndexInterface {
 public:
  SuffixArrayIndex() = default;

  // Initialize and construct the suffix array of the |text| string of length
  // |n|. The memory pointed by |text| must be kept alive. Returns whether the
  // construction succeeded.
  bool Init(const uint8_t* text, size_t n);

  // SuffixArrayIndexInterface overrides.
  void SearchPrefix(const uint8_t* target,
                    size_t length,
                    size_t* out_length,
                    uint64_t* out_pos) const override;

 private:
  const uint8_t* text_{nullptr};  // Owned by the caller.
  size_t n_{0};

  std::vector<SAIDX> sa_;
};

template <typename SAIDX>
bool SuffixArrayIndex<SAIDX>::Init(const uint8_t* text, size_t n) {
  if (!sa_.empty()) {
    // Already initialized.
    LOG(ERROR) << "SuffixArray already initialized";
    return false;
  }
  if (static_cast<uint64_t>(n) >
      static_cast<uint64_t>(std::numeric_limits<SAIDX>::max())) {
    LOG(ERROR) << "Input too big (" << n << ") for this implementation";
    return false;
  }
  text_ = text;
  n_ = n;
  sa_.resize(n + 1);

  if (n > 0 && CallDivSufSort(text_, sa_.data(), n) != 0) {
    LOG(ERROR) << "divsufsrot() failed";
    return false;
  }

  return true;
}

template <typename SAIDX>
void SuffixArrayIndex<SAIDX>::SearchPrefix(const uint8_t* target,
                                           size_t length,
                                           size_t* out_length,
                                           uint64_t* out_pos) const {
  SAIDX suf_left;
  SAIDX count =
      CallSaSearch(text_, n_, target, length, sa_.data(), n_, &suf_left);
  if (count > 0) {
    // This is the simple case where we found the whole |target| string was
    // found.
    *out_pos = sa_[suf_left];
    *out_length = length;
    return;
  }
  // In this case, |suf_left| points to the first suffix array position such
  // that the suffix at that position is lexicographically larger than |target|.
  // We only need to check whether the previous entry or the current entry is a
  // longer match.
  size_t prev_suffix_len = 0;
  if (suf_left > 0) {
    const size_t prev_max_len =
        std::min(n_ - static_cast<size_t>(sa_[suf_left - 1]), length);
    const uint8_t* prev_suffix = text_ + sa_[suf_left - 1];
    prev_suffix_len =
        std::mismatch(target, target + prev_max_len, prev_suffix).first -
        target;
  }

  size_t next_suffix_len = 0;
  if (static_cast<size_t>(suf_left) < n_) {
    const uint8_t* next_suffix = text_ + sa_[suf_left];
    const size_t next_max_len =
        std::min(n_ - static_cast<size_t>(sa_[suf_left]), length);
    next_suffix_len =
        std::mismatch(target, target + next_max_len, next_suffix).first -
        target;
  }

  *out_length = std::max(next_suffix_len, prev_suffix_len);
  if (!*out_length) {
    *out_pos = 0;
  } else if (next_suffix_len >= prev_suffix_len) {
    *out_pos = sa_[suf_left];
  } else {
    *out_pos = sa_[suf_left - 1];
  }
}

std::unique_ptr<SuffixArrayIndexInterface> CreateSuffixArrayIndex(
    const uint8_t* text,
    size_t n) {
  // The maximum supported size when using the suffix array based on the 32-bit
  // saidx_t type. We limit this to something a bit smaller (16 bytes smaller)
  // than the maximum representable number so references like "n + 1" are don't
  // overflow.
  const size_t kMaxSaidxSize = std::numeric_limits<saidx_t>::max() - 16;
  std::unique_ptr<SuffixArrayIndexInterface> ret;

  if (n > kMaxSaidxSize) {
    SuffixArrayIndex<saidx64_t>* sa_ptr = new SuffixArrayIndex<saidx64_t>();
    ret.reset(sa_ptr);
    if (!sa_ptr->Init(text, n))
      return nullptr;
  } else {
    SuffixArrayIndex<saidx_t>* sa_ptr = new SuffixArrayIndex<saidx_t>();
    ret.reset(sa_ptr);
    if (!sa_ptr->Init(text, n))
      return nullptr;
  }
  return ret;
}


}  // namespace bsdiff

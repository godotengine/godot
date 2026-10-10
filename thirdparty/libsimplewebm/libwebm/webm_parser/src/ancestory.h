// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_ANCESTORY_H_
#define SRC_ANCESTORY_H_

#include <cassert>
#include <cstddef>
#include <iterator>

#include "webm/id.h"

namespace webm {

// Represents an element's ancestory in descending order. For example, the
// Id::kTrackNumber element has an ancestory of {Id::kSegment, Id::kTracks,
// Id::kTrackEntry}.
class Ancestory {
 public:
  // Constructs an empty ancestory.
  Ancestory() = default;

  Ancestory(const Ancestory&) = default;
  Ancestory(Ancestory&&) = default;
  Ancestory& operator=(const Ancestory&) = default;
  Ancestory& operator=(Ancestory&&) = default;

  // Returns the ancestory with the top-level parent removed. For example, if
  // the current ancestory is {Id::kSegment, Id::kTracks, Id::kTrackEntry}, next
  // will return {Id::kTracks, Id::kTrackEntry}. This must not be called if the
  // ancestory is empty.
  Ancestory next() const {
    assert(begin_ < end_);
    Ancestory copy = *this;
    ++copy.begin_;
    return copy;
  }

  // Gets the Id of the top-level parent. For example, if the current ancestory
  // is {Id::kSegment, Id::kTracks, Id::kTrackEntry}, id will return
  // Id::kSegment. This must not be called if the ancestory is empty.
  Id id() const {
    assert(begin_ < end_);
    return *begin_;
  }

  // Returns true if the ancestory is empty.
  bool empty() const { return begin_ == end_; }

  // Looks up the ancestory of the given id. Returns true and sets ancestory if
  // the element's ancestory could be deduced. Global elements (i.e. Id::kVoid)
  // and unknown elements can't have their ancestory deduced.
  static bool ById(Id id, Ancestory* ancestory);

 private:
  // Constructs an Ancestory using the first count elements of ancestory.
  // ancestory must have static storage duration.
  template <std::size_t N>
  Ancestory(const Id (&ancestory)[N], std::size_t count)
      : begin_(ancestory), end_(ancestory + count) {
    assert(count <= N);
  }

  // The following invariants apply to begin_ and end_:
  // begin_ <= end_
  // (begin_ == end_) || (std::begin(kIds) <= begin_ && end_ <= std::end(kIds))

  // The beginning (inclusive) of the sequence of IDs in kIds that defines the
  // ancestory.
  const Id* begin_ = nullptr;

  // The ending (exclusive) of the sequence of IDs in kIds that defines the
  // ancestory.
  const Id* end_ = nullptr;
};

}  // namespace webm

#endif  // SRC_ANCESTORY_H_

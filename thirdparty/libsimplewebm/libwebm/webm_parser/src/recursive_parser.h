// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_RECURSIVE_PARSER_H_
#define SRC_RECURSIVE_PARSER_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "src/element_parser.h"
#include "webm/callback.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Lazily instantiates a parser of type T, and uses that parser to handle all
// parsing operations. The parser is allocated when Init is called. This class
// is intended to be used with recursive elements, where a parser needs to
// recursively instantiate parsers of the same type.
template <typename T>
class RecursiveParser : public ElementParser {
 public:
  explicit RecursiveParser(std::size_t max_recursion_depth = 25)
      : max_recursion_depth_(max_recursion_depth) {};

  RecursiveParser(RecursiveParser&&) = default;
  RecursiveParser& operator=(RecursiveParser&&) = default;

  RecursiveParser(const RecursiveParser&) = delete;
  RecursiveParser& operator=(const RecursiveParser&) = delete;

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

    if (max_recursion_depth_ == 0) {
      return Status(Status::kExceededRecursionDepthLimit);
    }

    if (!impl_) {
      impl_.reset(new T(max_recursion_depth_ - 1));
    }

    return impl_->Init(metadata, max_size);
  }

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override {
    assert(max_recursion_depth_ > 0);
    if (!impl_) {
      impl_.reset(new T(max_recursion_depth_ - 1));
    }

    impl_->InitAfterSeek(child_ancestory, child_metadata);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    assert(callback != nullptr);
    assert(reader != nullptr);
    assert(num_bytes_read != nullptr);
    assert(impl_ != nullptr);

    return impl_->Feed(callback, reader, num_bytes_read);
  }

  decltype(std::declval<T>().value()) value() const {
    assert(impl_ != nullptr);

    return impl_->value();
  }

  decltype(std::declval<T>().mutable_value()) mutable_value() {
    assert(impl_ != nullptr);

    return impl_->mutable_value();
  }

 private:
  std::unique_ptr<T> impl_;
  std::size_t max_recursion_depth_;
};

}  // namespace webm

#endif  // SRC_RECURSIVE_PARSER_H_

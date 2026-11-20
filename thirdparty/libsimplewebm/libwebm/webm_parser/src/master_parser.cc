// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/master_parser.h"

#include <cassert>
#include <cstdint>
#include <limits>

#include "src/element_parser.h"
#include "src/skip_callback.h"
#include "webm/element.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#EBML_ex
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown
Status MasterParser::Init(const ElementMetadata& metadata,
                          std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  InitSetup(metadata.header_size, metadata.size, metadata.position);

  if (metadata.size != kUnknownElementSize) {
    max_size_ = metadata.size;
  } else {
    max_size_ = max_size;
  }

  if (metadata.size == 0) {
    state_ = State::kEndReached;
  } else {
    state_ = State::kFirstReadOfChildId;
  }

  return Status(Status::kOkCompleted);
}

void MasterParser::InitAfterSeek(const Ancestory& child_ancestory,
                                 const ElementMetadata& child_metadata) {
  InitSetup(kUnknownHeaderSize, kUnknownElementSize, kUnknownElementPosition);
  max_size_ = std::numeric_limits<std::uint64_t>::max();

  if (child_ancestory.empty()) {
    child_metadata_ = child_metadata;
    auto iter = parsers_.find(child_metadata_.id);
    assert(iter != parsers_.end());
    child_parser_ = iter->second.get();
    state_ = State::kGettingAction;
  } else {
    child_metadata_.id = child_ancestory.id();
    child_metadata_.header_size = kUnknownHeaderSize;
    child_metadata_.size = kUnknownElementSize;
    child_metadata_.position = kUnknownElementPosition;

    auto iter = parsers_.find(child_metadata_.id);
    assert(iter != parsers_.end());
    child_parser_ = iter->second.get();
    child_parser_->InitAfterSeek(child_ancestory.next(), child_metadata);
    state_ = State::kReadingChildBody;
  }
}

Status MasterParser::Feed(Callback* callback, Reader* reader,
                          std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  Callback* const original_callback = callback;

  SkipCallback skip_callback;
  if (action_ == Action::kSkip) {
    callback = &skip_callback;
  }

  Status status;
  std::uint64_t local_num_bytes_read;
  while (true) {
    switch (state_) {
      case State::kFirstReadOfChildId: {
        // This separate case for the first read of the child ID is needed to
        // avoid potential bugs where calling Feed() twice in a row on an
        // unsized element at the end of the stream would return
        // Status::kOkCompleted instead of Status::kEndOfFile (since we convert
        // Status::kEndOfFile to Status::kOkCompleted when EOF is hit for an
        // unsized element after its children have been fully parsed). Once
        // the ID parser consumes > 0 bytes, this state must be exited.
        assert(child_parser_ == nullptr);
        assert(my_size_ == kUnknownElementSize || total_bytes_read_ < my_size_);
        child_metadata_.position = reader->Position();
        child_metadata_.header_size = 0;
        status = id_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        total_bytes_read_ += local_num_bytes_read;
        child_metadata_.header_size +=
            static_cast<std::uint32_t>(local_num_bytes_read);
        if (status.code == Status::kEndOfFile &&
            my_size_ == kUnknownElementSize && local_num_bytes_read == 0) {
          state_ = State::kEndReached;
        } else if (!status.ok()) {
          if (local_num_bytes_read > 0) {
            state_ = State::kFinishingReadingChildId;
          }
          return status;
        } else if (status.completed_ok()) {
          state_ = State::kReadingChildSize;
        } else {
          state_ = State::kFinishingReadingChildId;
        }
        continue;
      }

      case State::kFinishingReadingChildId: {
        assert(child_parser_ == nullptr);
        assert(my_size_ == kUnknownElementSize || total_bytes_read_ < my_size_);
        status = id_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        total_bytes_read_ += local_num_bytes_read;
        child_metadata_.header_size +=
            static_cast<std::uint32_t>(local_num_bytes_read);
        if (!status.completed_ok()) {
          return status;
        }
        state_ = State::kReadingChildSize;
        continue;
      }

      case State::kReadingChildSize: {
        assert(child_parser_ == nullptr);
        assert(total_bytes_read_ > 0);
        status = size_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        total_bytes_read_ += local_num_bytes_read;
        child_metadata_.header_size +=
            static_cast<std::uint32_t>(local_num_bytes_read);
        if (!status.completed_ok()) {
          return status;
        }
        child_metadata_.id = id_parser_.id();
        child_metadata_.size = size_parser_.size();
        state_ = State::kValidatingChildSize;
        continue;
      }

      case State::kValidatingChildSize: {
        assert(child_parser_ == nullptr);

        std::uint64_t byte_count = total_bytes_read_;
        if (child_metadata_.size != kUnknownElementSize) {
          byte_count += child_metadata_.size;
        }

        std::uint64_t byte_cap = max_size_;
        // my_size_ is <= max_size_ if it's known, so pick the smaller value.
        if (my_size_ != kUnknownElementSize) {
          byte_cap = my_size_;
        }

        if (byte_count > byte_cap) {
          return Status(Status::kElementOverflow);
        }

        auto iter = parsers_.find(child_metadata_.id);
        bool unknown_child = iter == parsers_.end();

        if (my_size_ == kUnknownElementSize && unknown_child) {
          // The end of an unsized master element is considered to be the first
          // instance of an element that isn't a known/valid child element.
          has_cached_metadata_ = true;
          state_ = State::kEndReached;
          continue;
        } else if (unknown_child &&
                   child_metadata_.size == kUnknownElementSize) {
          // We can't skip or otherwise handle unknown elements with an unknown
          // size.
          return Status(Status::kIndefiniteUnknownElement);
        }
        if (unknown_child) {
          child_parser_ = &unknown_parser_;
        } else {
          child_parser_ = iter->second.get();
        }
        state_ = State::kGettingAction;
        continue;
      }

      case State::kGettingAction: {
        assert(child_parser_ != nullptr);
        status = callback->OnElementBegin(child_metadata_, &action_);
        if (!status.completed_ok()) {
          return status;
        }

        if (action_ == Action::kSkip) {
          callback = &skip_callback;
          if (child_metadata_.size != kUnknownElementSize) {
            child_parser_ = &skip_parser_;
          }
        }
        state_ = State::kInitializingChildParser;
        continue;
      }

      case State::kInitializingChildParser: {
        assert(child_parser_ != nullptr);
        status =
            child_parser_->Init(child_metadata_, max_size_ - total_bytes_read_);
        if (!status.completed_ok()) {
          return status;
        }
        state_ = State::kReadingChildBody;
        continue;
      }

      case State::kReadingChildBody: {
        assert(child_parser_ != nullptr);
        status = child_parser_->Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        total_bytes_read_ += local_num_bytes_read;
        if (!status.completed_ok()) {
          return status;
        }
        state_ = State::kChildFullyParsed;
        continue;
      }

      case State::kChildFullyParsed: {
        assert(child_parser_ != nullptr);
        std::uint64_t byte_cap = max_size_;
        // my_size_ is <= max_size_ if it's known, so pick the smaller value.
        if (my_size_ != kUnknownElementSize) {
          byte_cap = my_size_;
        }

        if (total_bytes_read_ > byte_cap) {
          return Status(Status::kElementOverflow);
        } else if (total_bytes_read_ == byte_cap) {
          state_ = State::kEndReached;
          continue;
        }

        if (child_parser_->GetCachedMetadata(&child_metadata_)) {
          state_ = State::kValidatingChildSize;
        } else {
          state_ = State::kFirstReadOfChildId;
        }
        PrepareForNextChild();
        callback = original_callback;
        continue;
      }

      case State::kEndReached: {
        return Status(Status::kOkCompleted);
      }
    }
  }
}

bool MasterParser::GetCachedMetadata(ElementMetadata* metadata) {
  assert(metadata != nullptr);

  if (has_cached_metadata_) {
    *metadata = child_metadata_;
  }
  return has_cached_metadata_;
}

void MasterParser::InitSetup(std::uint32_t header_size,
                             std::uint64_t size_in_bytes,
                             std::uint64_t position) {
  PrepareForNextChild();
  header_size_ = header_size;
  my_size_ = size_in_bytes;
  my_position_ = position;
  total_bytes_read_ = 0;
  has_cached_metadata_ = false;
}

void MasterParser::PrepareForNextChild() {
  // Do not reset child_metadata_ here.
  id_parser_ = {};
  size_parser_ = {};
  child_parser_ = nullptr;
  action_ = Action::kRead;
}

}  // namespace webm

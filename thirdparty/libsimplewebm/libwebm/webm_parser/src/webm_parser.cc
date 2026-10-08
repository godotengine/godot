// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "webm/webm_parser.h"

#include <cassert>
#include <cstdint>

#include "src/ebml_parser.h"
#include "src/master_parser.h"
#include "src/segment_parser.h"
#include "src/unknown_parser.h"
#include "webm/element.h"

namespace webm {

// Parses WebM EBML documents (i.e. level-0 WebM elements).
class WebmParser::DocumentParser {
 public:
  // Resets the parser after a seek to a new position in the reader.
  void DidSeek() {
    PrepareForNextChild();
    did_seek_ = true;
    state_ = State::kBegin;
  }

  // Feeds the parser; will return Status::kOkCompleted when the reader returns
  // Status::kEndOfFile, but only if the parser has already completed parsing
  // its child elements.
  Status Feed(Callback* callback, Reader* reader) {
    assert(callback != nullptr);
    assert(reader != nullptr);

    Callback* const original_callback = callback;
    if (action_ == Action::kSkip) {
      callback = &skip_callback_;
    }

    Status status;
    std::uint64_t num_bytes_read;
    while (true) {
      switch (state_) {
        case State::kBegin: {
          child_metadata_.header_size = 0;
          child_metadata_.position = reader->Position();
          state_ = State::kReadingChildId;
          continue;
        }

        case State::kReadingChildId: {
          assert(child_parser_ == nullptr);
          status = id_parser_.Feed(callback, reader, &num_bytes_read);
          child_metadata_.header_size += num_bytes_read;
          if (!status.completed_ok()) {
            if (status.code == Status::kEndOfFile &&
                reader->Position() == child_metadata_.position) {
              state_ = State::kEndReached;
              continue;
            }
            return status;
          }
          state_ = State::kReadingChildSize;
          continue;
        }

        case State::kReadingChildSize: {
          assert(child_parser_ == nullptr);
          status = size_parser_.Feed(callback, reader, &num_bytes_read);
          child_metadata_.header_size += num_bytes_read;
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

          if (child_metadata_.id == Id::kSegment) {
            child_parser_ = &segment_parser_;
            did_seek_ = false;
            state_ = State::kGettingAction;
            continue;
          } else if (child_metadata_.id == Id::kEbml) {
            child_parser_ = &ebml_parser_;
            did_seek_ = false;
            state_ = State::kGettingAction;
            continue;
          }

          Ancestory ancestory;
          if (did_seek_ && Ancestory::ById(child_metadata_.id, &ancestory)) {
            assert(!ancestory.empty());
            assert(ancestory.id() == Id::kSegment ||
                   ancestory.id() == Id::kEbml);

            if (ancestory.id() == Id::kSegment) {
              child_parser_ = &segment_parser_;
            } else {
              child_parser_ = &ebml_parser_;
            }

            child_parser_->InitAfterSeek(ancestory.next(), child_metadata_);
            child_metadata_.id = ancestory.id();
            child_metadata_.header_size = kUnknownHeaderSize;
            child_metadata_.size = kUnknownElementSize;
            child_metadata_.position = kUnknownElementPosition;
            did_seek_ = false;
            action_ = Action::kRead;
            state_ = State::kReadingChildBody;
            continue;
          }

          if (child_metadata_.id == Id::kVoid) {
            child_parser_ = &void_parser_;
          } else {
            if (child_metadata_.size == kUnknownElementSize) {
              return Status(Status::kIndefiniteUnknownElement);
            }
            child_parser_ = &unknown_parser_;
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
            callback = &skip_callback_;
            if (child_metadata_.size != kUnknownElementSize) {
              child_parser_ = &skip_parser_;
            }
          }
          state_ = State::kInitializingChildParser;
          continue;
        }

        case State::kInitializingChildParser: {
          assert(child_parser_ != nullptr);
          status = child_parser_->Init(child_metadata_, child_metadata_.size);
          if (!status.completed_ok()) {
            return status;
          }
          state_ = State::kReadingChildBody;
          continue;
        }

        case State::kReadingChildBody: {
          assert(child_parser_ != nullptr);
          status = child_parser_->Feed(callback, reader, &num_bytes_read);
          if (!status.completed_ok()) {
            return status;
          }
          if (child_parser_->GetCachedMetadata(&child_metadata_)) {
            state_ = State::kValidatingChildSize;
          } else {
            child_metadata_.header_size = 0;
            state_ = State::kReadingChildId;
          }
          PrepareForNextChild();
          callback = original_callback;
          child_metadata_.position = reader->Position();
          continue;
        }

        case State::kEndReached: {
          return Status(Status::kOkCompleted);
        }
      }
    }
  }

 private:
  // Parsing states for the finite-state machine.
  enum class State {
    /* clang-format off */
    // State                      Transitions to state      When
    kBegin,                    // kReadingChildId           done
    kReadingChildId,           // kReadingChildSize         done
                               // kEndReached               EOF
    kReadingChildSize,         // kValidatingChildSize      done
    kValidatingChildSize,      // kGettingAction            done
    kGettingAction,            // kInitializingChildParser  done
    kInitializingChildParser,  // kReadingChildBody         done
    kReadingChildBody,         // kValidatingChildSize      cached metadata
                               // kReadingChildId           otherwise
    kEndReached,               // No transitions from here
    /* clang-format on */
  };

  // The parser for parsing child element Ids.
  IdParser id_parser_;

  // The parser for parsing child element sizes.
  SizeParser size_parser_;

  // The parser for Id::kEbml elements.
  EbmlParser ebml_parser_;

  // The parser for Id::kSegment child elements.
  SegmentParser segment_parser_;

  // The parser for Id::kVoid child elements.
  VoidParser void_parser_;

  // The parser used when skipping elements (if the element's size is known).
  SkipParser skip_parser_;

  // The parser used for unknown children.
  UnknownParser unknown_parser_;

  // The callback used when skipping elements.
  SkipCallback skip_callback_;

  // The parser that is parsing the current child element.
  ElementParser* child_parser_ = nullptr;

  // Metadata for the current child being parsed.
  ElementMetadata child_metadata_ = {};

  // Action for the current child being parsed.
  Action action_ = Action::kRead;

  // True if a seek was performed and the parser needs to handle it.
  bool did_seek_ = false;

  // The current state of the finite state machine.
  State state_ = State::kBegin;

  // Resets state in preparation for parsing a child element.
  void PrepareForNextChild() {
    id_parser_ = {};
    size_parser_ = {};
    child_parser_ = nullptr;
    action_ = Action::kRead;
  }
};

// We have to explicitly declare a destructor (even if it's just defaulted)
// because using the pimpl idiom with std::unique_ptr requires it. See Herb
// Sutter's GotW #100 for further explanation.
WebmParser::~WebmParser() = default;

WebmParser::WebmParser() : parser_(new DocumentParser) {}

void WebmParser::DidSeek() {
  parser_->DidSeek();
  parsing_status_ = Status(Status::kOkPartial);
}

Status WebmParser::Feed(Callback* callback, Reader* reader) {
  assert(callback != nullptr);
  assert(reader != nullptr);

  if (parsing_status_.is_parsing_error()) {
    return parsing_status_;
  }
  parsing_status_ = parser_->Feed(callback, reader);
  return parsing_status_;
}

void WebmParser::Swap(WebmParser* other) {
  assert(other != nullptr);
  parser_.swap(other->parser_);
  std::swap(parsing_status_, other->parsing_status_);
}

void swap(WebmParser& left, WebmParser& right) { left.Swap(&right); }

}  // namespace webm

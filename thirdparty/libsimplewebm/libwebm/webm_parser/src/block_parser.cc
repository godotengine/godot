// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/block_parser.h"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

#include "src/parser_utils.h"
#include "webm/element.h"

namespace webm {

namespace {

// The ParseBasicBlockFlags functions parse extra flag bits into the block,
// depending on the type of block that is being parsed.
void ParseBasicBlockFlags(std::uint8_t /* flags */, Block* /* block */) {
  // Block has no extra flags that aren't already handled.
}

void ParseBasicBlockFlags(std::uint8_t flags, SimpleBlock* block) {
  block->is_key_frame = (0x80 & flags) != 0;
  block->is_discardable = (0x01 & flags) != 0;
}

// The BasicBlockBegin functions call the Callback event handler and get the
// correct action for the parser, depending on the type of block that is being
// parsed.
Status BasicBlockBegin(const ElementMetadata& metadata, const Block& block,
                       Callback* callback, Action* action) {
  return callback->OnBlockBegin(metadata, block, action);
}

Status BasicBlockBegin(const ElementMetadata& metadata,
                       const SimpleBlock& block, Callback* callback,
                       Action* action) {
  return callback->OnSimpleBlockBegin(metadata, block, action);
}

// The BasicBlockEnd functions call the Callback event handler depending on the
// type of block that is being parsed.
Status BasicBlockEnd(const ElementMetadata& metadata, const Block& block,
                     Callback* callback) {
  return callback->OnBlockEnd(metadata, block);
}

Status BasicBlockEnd(const ElementMetadata& metadata, const SimpleBlock& block,
                     Callback* callback) {
  return callback->OnSimpleBlockEnd(metadata, block);
}

}  // namespace

template <typename T>
Status BasicBlockParser<T>::Init(const ElementMetadata& metadata,
                                 std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  if (metadata.size == kUnknownElementSize || metadata.size < 5) {
    return Status(Status::kInvalidElementSize);
  }

  *this = {};
  frame_metadata_.parent_element = metadata;

  return Status(Status::kOkCompleted);
}

template <typename T>
Status BasicBlockParser<T>::Feed(Callback* callback, Reader* reader,
                                 std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  Status status;
  std::uint64_t local_num_bytes_read;

  while (true) {
    switch (state_) {
      case State::kReadingHeader: {
        status = header_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        header_bytes_read_ += local_num_bytes_read;
        if (!status.completed_ok()) {
          return status;
        }
        value_.track_number = header_parser_.value().track_number;
        value_.timecode = header_parser_.value().timecode;

        std::uint8_t flags = header_parser_.value().flags;
        value_.is_visible = (0x08 & flags) == 0;
        value_.lacing = static_cast<Lacing>(flags & 0x06);
        ParseBasicBlockFlags(flags, &value_);

        if (value_.lacing == Lacing::kNone) {
          value_.num_frames = 1;
          state_ = State::kGettingAction;
        } else {
          state_ = State::kReadingLaceCount;
        }
        continue;
      }

      case State::kReadingLaceCount: {
        assert(lace_sizes_.empty());
        std::uint8_t lace_count;
        status = ReadByte(reader, &lace_count);
        if (!status.completed_ok()) {
          return status;
        }
        ++*num_bytes_read;
        ++header_bytes_read_;
        // Lace count is stored as (count - 1).
        value_.num_frames = lace_count + 1;
        state_ = State::kGettingAction;
        continue;
      }

      case State::kGettingAction: {
        Action action = Action::kRead;
        status = BasicBlockBegin(frame_metadata_.parent_element, value_,
                                 callback, &action);
        if (!status.completed_ok()) {
          return status;
        }
        if (action == Action::kSkip) {
          state_ = State::kSkipping;
        } else if (value_.lacing == Lacing::kNone || value_.num_frames == 1) {
          state_ = State::kValidatingSize;
        } else if (value_.lacing == Lacing::kXiph) {
          state_ = State::kReadingXiphLaceSizes;
        } else if (value_.lacing == Lacing::kEbml) {
          state_ = State::kReadingFirstEbmlLaceSize;
        } else {
          state_ = State::kCalculatingFixedLaceSizes;
        }
        continue;
      }

      case State::kReadingXiphLaceSizes:
        assert(value_.num_frames > 0);
        while (static_cast<int>(lace_sizes_.size()) < value_.num_frames - 1) {
          std::uint8_t byte;
          do {
            status = ReadByte(reader, &byte);
            if (!status.completed_ok()) {
              return status;
            }
            ++*num_bytes_read;
            ++header_bytes_read_;
            xiph_lace_size_ += byte;
          } while (byte == 255);

          lace_sizes_.push_back(xiph_lace_size_);
          xiph_lace_size_ = 0;
        }
        state_ = State::kValidatingSize;
        continue;

      case State::kReadingFirstEbmlLaceSize:
        assert(value_.num_frames > 0);
        assert(lace_sizes_.empty());
        status = uint_parser_.Feed(callback, reader, &local_num_bytes_read);
        *num_bytes_read += local_num_bytes_read;
        header_bytes_read_ += local_num_bytes_read;
        if (!status.completed_ok()) {
          return status;
        }
        lace_sizes_.push_back(uint_parser_.value());
        uint_parser_ = {};
        state_ = State::kReadingEbmlLaceSizes;
        continue;

      case State::kReadingEbmlLaceSizes:
        assert(value_.num_frames > 0);
        assert(!lace_sizes_.empty());
        while (static_cast<int>(lace_sizes_.size()) < value_.num_frames - 1) {
          status = uint_parser_.Feed(callback, reader, &local_num_bytes_read);
          *num_bytes_read += local_num_bytes_read;
          header_bytes_read_ += local_num_bytes_read;
          if (!status.completed_ok()) {
            return status;
          }
          constexpr std::uint64_t one = 1;  // Prettier than a static_cast.
          std::uint64_t offset =
              (one << (uint_parser_.encoded_length() * 7 - 1)) - 1;
          lace_sizes_.push_back(lace_sizes_.back() + uint_parser_.value() -
                                offset);
          uint_parser_ = {};
        }
        state_ = State::kValidatingSize;
        continue;

      case State::kCalculatingFixedLaceSizes: {
        assert(value_.num_frames > 0);
        assert(lace_sizes_.empty());
        if (header_bytes_read_ >= frame_metadata_.parent_element.size) {
          return Status(Status::kInvalidElementValue);
        }
        std::uint64_t laced_data_size =
            frame_metadata_.parent_element.size - header_bytes_read_;
        std::uint64_t frame_size = laced_data_size / value_.num_frames;
        if (laced_data_size % value_.num_frames != 0) {
          return Status(Status::kInvalidElementValue);
        }
        lace_sizes_.resize(value_.num_frames, frame_size);
        frame_metadata_.position =
            frame_metadata_.parent_element.position + header_bytes_read_;
        frame_metadata_.size = frame_size;
        state_ = State::kReadingFrames;
        continue;
      }

      case State::kValidatingSize: {
        std::uint64_t sum = std::accumulate(
            lace_sizes_.begin(), lace_sizes_.end(), header_bytes_read_);
        if (sum >= frame_metadata_.parent_element.size) {
          return Status(Status::kInvalidElementValue);
        }
        lace_sizes_.push_back(frame_metadata_.parent_element.size - sum);
        frame_metadata_.position = frame_metadata_.parent_element.position +
                                   frame_metadata_.parent_element.header_size +
                                   header_bytes_read_;
        frame_metadata_.size = lace_sizes_[0];
        state_ = State::kReadingFrames;
        continue;
      }

      case State::kSkipping:
        do {
          // Skip the remaining part of the header and all of the frames.
          status = reader->Skip(
              frame_metadata_.parent_element.size - header_bytes_read_,
              &local_num_bytes_read);
          *num_bytes_read += local_num_bytes_read;
          header_bytes_read_ += local_num_bytes_read;
        } while (status.code == Status::kOkPartial);
        return status;

      case State::kReadingFrames:
        assert(value_.num_frames > 0);
        assert(static_cast<int>(lace_sizes_.size()) == value_.num_frames);
        for (; current_lace_ < lace_sizes_.size(); ++current_lace_) {
          const std::uint64_t original = lace_sizes_[current_lace_];
          status = callback->OnFrame(frame_metadata_, reader,
                                     &lace_sizes_[current_lace_]);
          *num_bytes_read += original - lace_sizes_[current_lace_];
          if (!status.completed_ok()) {
            return status;
          }
          assert(lace_sizes_[current_lace_] == 0);
          if (current_lace_ + 1 < lace_sizes_.size()) {
            frame_metadata_.position += frame_metadata_.size;
            frame_metadata_.size = lace_sizes_[current_lace_ + 1];
          }
        }
        state_ = State::kDone;
        continue;

      case State::kDone:
        return BasicBlockEnd(frame_metadata_.parent_element, value_, callback);
    }
  }
}

template <typename T>
bool BasicBlockParser<T>::WasSkipped() const {
  return state_ == State::kSkipping;
}

template class BasicBlockParser<Block>;
template class BasicBlockParser<SimpleBlock>;

}  // namespace webm

// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_AUDIO_PARSER_H_
#define SRC_AUDIO_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/float_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/callback.h"
#include "webm/dom_types.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Audio
// http://www.webmproject.org/docs/container/#Audio
class AudioParser : public MasterValueParser<Audio> {
 public:
  AudioParser()
      : MasterValueParser<Audio>(
            MakeChild<FloatParser>(Id::kSamplingFrequency,
                                   &Audio::sampling_frequency),
            MakeChild<FloatParser>(Id::kOutputSamplingFrequency,
                                   &Audio::output_frequency)
                .NotifyOnParseComplete(),
            MakeChild<UnsignedIntParser>(Id::kChannels, &Audio::channels),
            MakeChild<UnsignedIntParser>(Id::kBitDepth, &Audio::bit_depth)) {}

  Status Init(const ElementMetadata& metadata,
              std::uint64_t max_size) override {
    output_frequency_has_value_ = false;

    return MasterValueParser::Init(metadata, max_size);
  }

  void InitAfterSeek(const Ancestory& child_ancestory,
                     const ElementMetadata& child_metadata) override {
    output_frequency_has_value_ = false;

    return MasterValueParser::InitAfterSeek(child_ancestory, child_metadata);
  }

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override {
    const Status status =
        MasterValueParser::Feed(callback, reader, num_bytes_read);
    if (status.completed_ok()) {
      FixMissingOutputFrequency();
    }
    return status;
  }

 protected:
  void OnChildParsed(const ElementMetadata& metadata) override {
    assert(metadata.id == Id::kOutputSamplingFrequency);

    output_frequency_has_value_ = metadata.size > 0;
  }

 private:
  bool output_frequency_has_value_;

  void FixMissingOutputFrequency() {
    if (!output_frequency_has_value_) {
      *mutable_value()->output_frequency.mutable_value() =
          value().sampling_frequency.value();
    }
  }
};

}  // namespace webm

#endif  // SRC_AUDIO_PARSER_H_
